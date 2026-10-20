"""SE(3), weighted correspondences, and first-order left-error filtering."""
import torch


def skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def eye(n, ref):
    return torch.eye(n, dtype=ref.dtype, device=ref.device).expand(*ref.shape[:-2], n, n)


def exp(x):
    a = x.new_zeros(*x.shape[:-1], 4, 4)
    a[..., :3, :3] = skew(x[..., 3:])
    a[..., :3, 3] = x[..., :3]
    return torch.matrix_exp(a)


def log(t):
    r = t[..., :3, :3]
    v = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                     r[..., 1, 0]-r[..., 0, 1]), -1)
    s = (.25*v.square().sum(-1)+1e-16).sqrt()
    c = ((r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    angle = torch.atan2(s, c)
    w = v*(angle/(2*s))[..., None]
    # Antisymmetric entries vanish at pi: recover axis from symmetric entries.
    diagonal = ((r.diagonal(dim1=-2, dim2=-1)+1)/2).clamp_min(1e-12).sqrt()
    a0, a1, a2 = diagonal.unbind(-1)
    xy, xz, yz = r[..., 0, 1]+r[..., 1, 0], r[..., 0, 2]+r[..., 2, 0], r[..., 1, 2]+r[..., 2, 1]
    candidates = torch.stack((torch.stack((a0, xy/(4*a0), xz/(4*a0)), -1),
                              torch.stack((xy/(4*a1), a1, yz/(4*a1)), -1),
                              torch.stack((xz/(4*a2), yz/(4*a2), a2), -1)), -2)
    index = diagonal.argmax(-1)[..., None, None].expand(*diagonal.shape[:-1], 1, 3)
    axis = candidates.gather(-2, index).squeeze(-2)
    axis = axis/axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sign = torch.where((axis*v).sum(-1, keepdim=True) < 0, -1., 1.)
    w = torch.where((c < -.9999)[..., None], axis*sign*angle[..., None], w)
    a2 = w.square().sum(-1)[..., None, None]
    a = (a2+1e-16).sqrt()
    b = torch.where(a2 < 1e-6, .5-a2/24+a2.square()/720, (1-a.cos())/a2.clamp_min(1e-16))
    d = torch.where(a2 < 1e-6, 1/6-a2/120+a2.square()/5040, (a-a.sin())/(a2*a).clamp_min(1e-16))
    k = skew(w)
    jac = eye(3, k)+b*k+d*(k@k)
    rho = torch.linalg.solve(jac, t[..., :3, 3, None]).squeeze(-1)
    return torch.cat((rho, w), -1)


def inv(t):
    r = t[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(t)
    out[..., :3, :3] = r
    out[..., :3, 3] = -(r@t[..., :3, 3, None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def transform(t, p):
    return p@t[..., :3, :3].transpose(-1, -2)+t[..., None, :3, 3]


def adjoint(t):
    r = t[..., :3, :3]
    z = torch.zeros_like(r)
    return torch.cat((torch.cat((r, skew(t[..., :3, 3])@r), -1), torch.cat((z, r), -1)), -2)


def algebra(x):
    w, v = skew(x[..., 3:]), skew(x[..., :3])
    z = torch.zeros_like(w)
    return torch.cat((torch.cat((w, v), -1), torch.cat((z, w), -1)), -2)


def left_jacobian(x):
    """Convergent series; intended for local inter-frame increments."""
    a = algebra(x)
    out = eye(6, a).clone()
    term = out
    for n in range(1, 13):
        term = term@a/(n+1)
        out = out+term
    return out


def rigid(src, dst, weights):
    """Known weighted 3D pairs, not averaged candidate coordinates."""
    w = weights/weights.sum(-1, keepdim=True).clamp_min(1e-8)
    a, b = (src*w[..., None]).sum(-2), (dst*w[..., None]).sum(-2)
    h = (src-a[..., None, :]).transpose(-1, -2)@((dst-b[..., None, :])*w[..., None])
    # Tiny distinct jitter protects SVD backward at repeated singular values.
    h = h+torch.diag(h.new_tensor([1e-8, 2e-8, 3e-8]))
    u, singular, vh = torch.linalg.svd(h)
    v = vh.transpose(-1, -2)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.linalg.det(v@u.transpose(-1, -2)).detach()
    r = (v*correction[..., None, :])@u.transpose(-1, -2)
    out = eye(4, h).clone()
    out[..., :3, :3] = r
    out[..., :3, 3] = b-(r@a[..., None]).squeeze(-1)
    error = (transform(out, src)-dst).square().sum(-1)
    return out, (error*w).sum(-1), singular


def predict(state, q, dt):
    c, o, vc, vo, p = (state[k] for k in ('camera', 'object', 'vc', 'vo', 'cov'))
    dc, do = vc*dt, vo*dt
    tc, to = exp(dc), exp(do)
    f = eye(24, p).clone()
    f[:, :6, :6] = adjoint(tc)
    f[:, :6, 6:12] = left_jacobian(dc)*dt
    f[:, 12:18, 12:18] = adjoint(to)
    f[:, 12:18, 18:24] = left_jacobian(do)*dt
    pp = f@p@f.transpose(-1, -2)+q*dt
    return dict(camera=tc@c, object=to@o, vc=vc, vo=vo, cov=pp)


def observation_jacobian(camera):
    h = camera.new_zeros(camera.shape[0], 12, 24)
    h[:, :6, :6] = torch.eye(6, device=camera.device, dtype=camera.dtype)
    a = adjoint(inv(camera))
    h[:, 6:, :6], h[:, 6:, 12:18] = -a, a
    return h


def correct(state, camera_measurement, object_in_camera, measurement_cov):
    c, o, p = state['camera'], state['object'], state['cov']
    residual = torch.cat((log(camera_measurement@inv(c)),
                          log(object_in_camera@inv(inv(c)@o))), -1)
    h = observation_jacobian(c)
    s = h@p@h.transpose(-1, -2)+measurement_cov
    gain = torch.linalg.solve(s, h@p).transpose(-1, -2)
    delta = (gain@residual[..., None]).squeeze(-1)
    a = eye(24, p)-gain@h
    posterior = a@p@a.transpose(-1, -2)+gain@measurement_cov@gain.transpose(-1, -2)
    reset = eye(24, p).clone()
    reset[:, :6, :6] += .5*algebra(delta[:, :6])
    reset[:, 12:18, 12:18] += .5*algebra(delta[:, 12:18])
    posterior = reset@posterior@reset.transpose(-1, -2)
    posterior = .5*(posterior+posterior.transpose(-1, -2))+eye(24, p)*1e-8
    return dict(camera=exp(delta[:, :6])@c, object=exp(delta[:, 12:18])@o,
                vc=state['vc']+delta[:, 6:12], vo=state['vo']+delta[:, 18:24], cov=posterior), residual
