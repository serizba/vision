import numpy as np
import numba as nb

def psnr(u, I):
    return  10 * np.log((I.size * (I.max() - I.min())) / ((I - u)**2).sum())

def div(n):
    ny, _ = np.gradient(n[..., 0])
    _, nx = np.gradient(n[..., 1])
    return ny + nx

@nb.njit
def flow2color(x, y):
  radius = np.sqrt(x * x + y * y);
  radius = min(max(radius, 0.0), 1.0)

  if x == 0.0:
    if y >= 0.0:
      phi = 0.5 * np.pi
    else:
      phi = 1.5 * np.pi
  elif x > 0.0:
    if y >= 0.0:
      phi = np.arctan(y/x);
    else:
      phi = 2.0 * np.pi + np.arctan(y/x)
  else:
    phi = np.pi + np.arctan(y/x)
 
  phi *= 0.5

  if ((phi >= 0.0) and (phi < 0.125 * np.pi)):
    beta  = phi / (0.125 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha * 255.0 + beta * 255.0));
    G = int(radius * (alpha *   0.0 + beta *   0.0));
    B = int(radius * (alpha *   0.0 + beta * 255.0));

  if ((phi >= 0.125 * np.pi) and (phi < 0.25 * np.pi)):
    beta  = (phi-0.125 * np.pi) / (0.125 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha * 255.0 + beta *  64.0));
    G = int(radius * (alpha *   0.0 + beta *  64.0));
    B = int(radius * (alpha * 255.0 + beta * 255.0));

  if ((phi >= 0.25 * np.pi) and (phi < 0.375 * np.pi)):
    beta  = (phi - 0.25 * np.pi) / (0.125 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha *  64.0 + beta *   0.0));
    G = int(radius * (alpha *  64.0 + beta * 255.0));
    B = int(radius * (alpha * 255.0 + beta * 255.0));

  if ((phi >= 0.375 * np.pi) and (phi < 0.5 * np.pi)):
    beta  = (phi - 0.375 * np.pi) / (0.125 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha *   0.0 + beta *   0.0));
    G = int(radius * (alpha * 255.0 + beta * 255.0));
    B = int(radius * (alpha * 255.0 + beta *   0.0));

  if ((phi >= 0.5 * np.pi) and (phi < 0.75 * np.pi)):
    beta  = (phi - 0.5 * np.pi) / (0.25 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha *   0.0 + beta * 255.0));
    G = int(radius * (alpha * 255.0 + beta * 255.0));
    B = int(radius * (alpha *   0.0 + beta *   0.0));

  if ((phi >= 0.75 * np.pi) and (phi <= np.pi)):
    beta  = (phi - 0.75 * np.pi) / (0.25 * np.pi);
    alpha = 1.0 - beta;
    R = int(radius * (alpha * 255.0 + beta * 255.0));
    G = int(radius * (alpha * 255.0 + beta *   0.0));
    B = int(radius * (alpha *   0.0 + beta *   0.0));

  return min(max(R, 0.0), 255.0), min(max(G, 0.0), 255.0), min(max(B, 0.0), 255.0)

@nb.njit
def flow2image(flow):
    res = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            res[i,j] = np.uint8(flow2color(flow[i,j,0], flow[i,j,1]))
    return res

def epe(gt, y):
    return np.sqrt((y[:, 0]-gt[:,0])**2 + (y[:, 1]-gt[:,1])**2).mean()
def aae(gt, y):
    return np.rad2deg(np.arccos((y[:, 0] * gt[:, 0] + y[:, 1] * gt[:, 1] + 1) / np.sqrt((y[:, 0]**2 + y[:, 1]**2 + 1) * (gt[:, 0]**2 + gt[:, 1]**2 + 1))).mean())
