import numpy as np
import hyperjet as hj

def compute(x):
  xa,ya, xb,yb, xc,yc, xd,yd, xe,ye, xf,yf, fxa,fya, fxb,fyb, fxc,fyc, fxd,fyd, fxe,fye, fxf,fyf, s1, s2, s3, s4, s5 = \
    hj.variables(x)

  a = np.array([xa, ya])
  b = np.array([xb, yb])
  c = np.array([xc, yc])
  d = np.array([xd, yd])
  e = np.array([xe, ye])
  f = np.array([xf, yf])

  def distance(a, b):
    v = b - a
    return np.dot(v, v)**0.5

  def direction(a, b):
    v = b - a
    l = np.dot(v, v)**0.5
    return v / l

  l1 = distance(a, b)
  l2 = distance(b, c)
  l3 = distance(c, d)
  l4 = distance(d, e)
  l5 = distance(e, f)

  p = sum([
    s1 * (l1 - 1),
    s2 * (l2 - 1),
    s3 * (l3 - 1),
    s4 * (l4 - 1),
    s5 * (l5 - 1),
    -(xa - xa.f) * fxa,
    # ya * fya,
    # xb * fxb,
    # yb * fyb,
    # xc * fxc,
    # yc * fyc,
    # xd * fxd,
    # yd * fyd,
    # xe * fxe,
    # ye * fye,
    -(xf - xf.f) * fxf,
    # yf * fyf,
    # (fxb - 0)**2,
    # (fyb - 1)**2,
    # (fxc - 0)**2,
    # (fyc - 1)**2,
    # (fxd - 0)**2,
    # (fyd - 1)**2,
    # (fxe - 0)**2,
    # (fye - 1)**2,
    # (l1 - 2)**2,
    # (l2 - 2)**2,
    # (l4 - 2)**2,
    # (l5 - 2)**2,
    # (xc - 2)**2,
    # (yc - 0)**2,
    # (xd - 3)**2,
    # (yd - 0)**2,
  ])

  return p

x = np.array([
  0,0, 1,0, 2,0, 3,0, 4,0, 5,0,
  -1,0, 0,0, 0,0, 0,0, 0,0, 1,0,
  1, 1, 1, 1, 1,
], float)



for i in range(1):
  p = compute(x)

  r = p.g

  print(np.linalg.norm(r))

  k = p.hm()
  k += np.eye(len(k)) * 1e-6

  x -= np.linalg.solve(k, r)

# print(x)