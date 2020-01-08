import numpy as np

def flow2color(x, y):
  radius = np.sqrt(x * x + y * y);
  radius = np.clip(radius, 0.0, 1.0)

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

  return np.clip([R,G,B], 0.0, 255.0)