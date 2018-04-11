import re
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

print("Opening file..")
file_name = "test_gmm_4.txt"
file = open(file_name, "r")
blob = file.read()
file.close()

list_points = []
lines = blob.split("\n")

print("Translating coordinates")
for line in lines:
  if line :
    [x,y] = line.replace("[","").replace("]", "").split(",")
    list_points.append(np.array([float(x),float(y)]))

k = int(input("How many clusters?\n"))

points = np.asarray(list_points)
c_points = len(points)

mus = list(np.random.uniform(-500, 501, size=(k, 2)))

lambdas = [1./k] * k

sigmas = [ np.identity(2) * 10000 for i in range(0, k) ]

r = lambda: np.random.randint(0, 256)
colors = [ '#%02X%02X%02X' % (r(),r(),r()) for i in range(0, k) ]

change = True

def gauss_pdf(point, mu, sigma):
  num = np.exp(-(1/2) * np.dot(np.dot(np.matrix(point - mu).T, np.linalg.inv(sigma)), np.matrix(point - mu)))
  den = np.sqrt(2*np.pi*np.linalg.det(sigma))
  return num/den

def latent_var(point, k, mus, lambdas, sigmas):
  p = multivariate_normal.pdf(x=point, mean=mus[k], cov=sigmas[k], allow_singular=True) * lambdas[k]
  normalize = sum([ np.multiply(multivariate_normal.pdf(x=point, mean=mus[i], cov=sigmas[i], allow_singular=True), lambdas[i]) for i in range(0, len(mus)) ])
  return p/normalize

def update_mu(k, points, mus, lambdas, sigmas, beta):
  num = sum([ latent_var(point, k, mus, lambdas, sigmas) * point for point in points ])
  return num/beta

def update_lambda(points, beta):
  return beta/c_points

def update_sigma(k, points, mus, lambdas, sigmas, beta):
  num = 0
  for point in points:
    num += np.multiply(
      np.multiply(
        latent_var(point, k, mus, lambdas, sigmas),
        np.matrix(point - mus[k]).T
      ), 
        np.matrix(point - mus[k])
      )
  return num/beta

def expectation(points, mus, lambdas, sigmas):
  cluster_type = []
  print(c_points)
  print(len(points))
  for i in range(0, c_points):
    pcluster = [ latent_var(points[i], j, mus, lambdas, sigmas) for j in range(0, k) ]
    cluster_type.append(pcluster.index(max(pcluster)))

  return cluster_type

def maximization(points, mus, lambdas, sigmas):
  nmus = mus[:]
  nlambdas = lambdas[:]
  nsigmas = sigmas[:]

  for i in range(0, k):
    beta = sum([ latent_var(point, i, mus, lambdas, sigmas) for point in points ])
    nmus[i] = update_mu(i, points, mus, lambdas, sigmas, beta)
    nlambdas[i] = update_lambda(points, beta)
    nsigmas[i] = update_sigma(i, points, mus, lambdas, sigmas, beta)

  return nmus, nlambdas, nsigmas 

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

ite = 0
print("Starting program")
while ite < 100:
  ite += 1
  #E
  #cluster_type = expectation(points, mus, lambdas, sigmas)

  #M
  nmus, nlambdas, nsigmas = maximization(points, mus, lambdas, sigmas)
  for i in range(0, len(mus)):
    change = not np.array_equal(mus[i], nmus[i])
  for i in range(0, len(lambdas)):
    change = not np.array_equal(lambdas[i], nlambdas[i])

  mus = nmus
  lambdas = nlambdas
  sigmas = nsigmas
  print("Ite", ite)

print("Results: ")
print("Ite", ite)
print("Mus", mus)
print("lambdas", lambdas)
print("sigmas", sigmas)

fig, ax = plt.subplots()

cluster_type = expectation(points, mus, lambdas, sigmas)

for i in range(0, c_points):
  [xp, yp] = points[i]
  ax.scatter(xp, yp, color=colors[cluster_type[i]])

for i in range(0, len(mus)):
  [xm, ym] = mus[i]
  cov = sigmas[i]
  vals, vecs = eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  for j in range(0, 6):
    w, h = j * np.sqrt(vals)
    ell = Ellipse(xy=(xm, ym), width=w, height=h, angle=theta, color=colors[i])
    ell.set_facecolor('none')
    ax.add_artist(ell)
  ax.scatter(xm, ym, marker="^", color=colors[i])

def calc_p():
  exit = input("Type \'exit\' to close. Calculate new point as \'x,y\' ")
  while exit != "exit":
    [x, y] = exit.split(",")
    ppoint = np.array([[x, y]])
    cluster_type = expectation(ppoint, mus, lambdas, sigmas)
    
    print("Mu", mus[cluster_type[0]])
    print("lambdas", lambdas[cluster_type[0]])
    print("sigmas", sigmas[cluster_type[0]])
    
    exit = input("Type \'exit\' to close. Calculate new point as \'x,y\' ")


t = threading.Thread(target=calc_p)
t.start()

plt.show()