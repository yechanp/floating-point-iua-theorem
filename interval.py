# import torch
# from torch.autograd import grad
import numpy as np
import itertools
import time
import argparse

# float16 basic functions

def square(x):
   r= x**2
   return np.array(r,np.float16)
def sin(x): # target(x) = sin( 10 pi x ) /2 
   x = np.array(x, np.float64)
   r= np.sin(10*np.pi*x)/2
   return np.array(r,np.float16)

################################ domain ################################

domain = [0,1] # codomain: [0,1]  [0,2**(-14)]
all_fp16 = np.arange(2**16, dtype=np.uint16).view(np.float16)
codomain_range= all_fp16[(all_fp16 >= 0) & (all_fp16 <=1 ) & np.isfinite(all_fp16)]
codomain_range= np.sort(codomain_range)

domain_range_zero_one = all_fp16[(all_fp16 >= 0) & (all_fp16 <=1) & np.isfinite(all_fp16)]
domain_range_zero_one= np.sort(domain_range_zero_one)
domain_range_zero_one = [np.array(item,np.float16) for item in domain_range_zero_one]


################################ domain ################################




def next_float(x):
  return np.nextafter(x, np.float16(max_float))
def before_float(x):
  return np.nextafter(x, np.float16(-max_float))
def sigma(x):
    return np.max(x,0)
def relu(x):
    return np.max(x,0)
    
# float16 basic information

M=10
emin = -14
tmp = np.array(2**(-M),dtype=np.float16)
eps = np.array(2**(-M-1),dtype=np.float16)
max_float = np.array(np.finfo(np.float16).max , dtype= np.float16)
zero_float = np.array(0.0 , dtype= np.float16)
omega =  np.nextafter(np.float16(0), np.float16(1))
omega_plus = next_float(omega)
one = np.array(1.0 , dtype= np.float16)
zero = zero_float

# Specific config
K = np.array(1.0, dtype=np.float16)
z= np.array(0.5, dtype=np.float16)
z_plus  = next_float(z)
z2      = next_float(z_plus)
z2_plus =   next_float(z2)
eta = np.array(3.0 ,dtype=np.float16)
eta_plus = np.nextafter(eta, np.float16(max_float))

c1 = np.array(1.0,np.float16)
c2 = np.array(0.0,np.float16)

#======================
# _Intvl = {l : torch.Tensor, r : torch.Tensor}
# _intvl_*: _Intvl^k -> _Intvl
#   x, y, W, b: _Intvl
#======================
class _Intvl:
    def __init__(self, l, r):
        self.l = l
        self.r = r
    def __repr__(self):
      return '{} {}'.format(self.l,self.r)
    def __eq__(self,other):
      return self.l == other.l and self.r == other.r

def _intvl_add(x, y):
    zl = x.l + y.l
    zr = x.r + y.r
    return _Intvl(zl, zr)

def _intvl_mul(x, y):
    zs = np.array(
        [x.l * y.l,
         x.l * y.r,
         x.r * y.l,
         x.r * y.r])
    zl = np.min(zs)
    zr = np.max(zs)
    return _Intvl(zl, zr)

def _intvl_sum(x):
    n = x.l.shape[0]
    res = _Intvl(torch.zeros(1), torch.zeros(1))
    for i in range(n):
        res = _intvl_add(res, _Intvl(x.l[i], x.r[i]))
    return res

def _intvl_affine(x, W, b):
    m = b.l.shape[0]
    res = _Intvl(torch.zeros(m), torch.zeros(m))
    for i in range(m):
        resi =\
            _intvl_add(_intvl_sum(
                _intvl_mul(x, _Intvl(W.l[i], W.r[i]))
            ), _Intvl(b.l[i], b.r[i]))
        res.l[i] = resi.l
        res.r[i] = resi.r
    return res

def _intvl_relu(x):
    zl = relu(x.l)
    zr = relu(x.r)
    return _Intvl(zl, zr)

class Intvl:
    def __init__(self, l, r):
        # l, r: torch.Tensor
        self.l = l
        self.r = r

    def affine(self, W, b):

        res = _intvl_affine(_Intvl(self.l, self.r),
                            _Intvl(W, W),
                            _Intvl(b, b))
        return Intvl(res.l, res.r)

    def relu(self):
        res = _intvl_relu(_Intvl(self.l, self.r))
        return Intvl(res.l, res.r)
# Decompose floating-point number into sign,exponent,significand
def decompose_normalized(x):
    # Convert input to np.float16 and get its raw bit representation.
    x = np.float16(x)
    bits = x.view(np.uint16)

    # For float16:
    #   1 sign bit (bit 15),
    #   5 exponent bits (bits 10-14),
    #   10 fraction bits (bits 0-9).
    sign = (bits >> 15) & 0x1
    exponent_raw = (bits >> 10) & 0x1F   # 5 bits for exponent
    fraction = bits & 0x3FF              # 10 bits for fraction

    # print(f"Value: {x}")
    # print(f"Raw bits: {bits:016b}")
    # print(f"Sign: {sign}")
    # print(f"Exponent bits: {exponent_raw:05b} (raw value: {exponent_raw})")
    # print(f"Fraction bits: {fraction:010b} (raw value: {fraction})")

    # For normalized numbers (exponent_raw != 0):
    if exponent_raw != 0:
        # The actual exponent is stored exponent minus the bias (15).
        exponent_actual = exponent_raw - 15
        # The significand (mantissa) includes an implicit 1, so it is in [1,2)
        significand = 1 + fraction / 1024

        exponent_adjusted = exponent_actual
        m = significand
        # print("\nNormalized number:")
        # print(f"  Actual exponent: {exponent_actual}")
        # print(f"  Actual significand: {significand}")
        # print(f"Representation: (-1)^({sign}) * 2^({exponent_actual}) * {significand}")
    else:
        # Special case: zero (both exponent and fraction are 0)
        if fraction == 0:
            exponent_adjusted = -14
            m = fraction
            pass
            # print("\nZero value.")
        else:
            # For subnormals, the stored exponent is 0.
            # Their value is computed as: x = 2^(-14) * (fraction / 1024).
            # To express x in the normalized form x = 2^E * m with m in [1,2),
            # we “shift” the fraction until it is in the form 1.xxx (i.e. m in [1,2)).
            # The initial effective exponent for subnormals is fixed at -14.
            m = fraction / 1024  # this is < 1
            shift = 0
            # Multiply m by 2 until it reaches [1,2)
            while m < 1 and shift < 10:
                m *= 2
                shift += 1
            # The adjusted exponent decreases by the number of shifts.
            exponent_adjusted = -14 - shift
    sign = -2 * sign + 1

    return sign, exponent_adjusted, m

sign_z, expo_z , sig_z       = decompose_normalized(z)
sign_eta, expo_eta , sig_eta = decompose_normalized(eta)

ivl_z = _Intvl(z,z)
ivl_z_plus = _Intvl(z_plus,z_plus)

ivl_eta = _Intvl(eta,eta)
ivl_eta_plus = _Intvl(eta_plus,eta_plus)

ivl_sigma_eta = _Intvl(sigma(eta),sigma(eta))
ivl_sigma_eta_plus = _Intvl(sigma(eta_plus),sigma(eta_plus))

ivl_zero = _Intvl(zero,zero)
ivl_one  = _Intvl(one,one)


ivl_max_float = _Intvl(max_float,max_float)
ivl_omega = _Intvl(omega,omega)
ivl_omega_plus = _Intvl(omega_plus,omega_plus)

def output_case1(z,eta):
   sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
   sign_z, expo_z , sig_z       = decompose_normalized(z)
   c_z = np.array(max(emin-expo_z,0), np.float16)
   sig_eta = np.array(sig_eta,dtype=np.float16)
   sig_z   = np.array(sig_z,dtype=np.float16)

   w = np.float_power(2.,-expo_z + expo_eta - c_z)
   b = (sig_eta - (sig_z * 2**(-c_z) ) ) * 2**(expo_eta)
   w = np.array(w,dtype=np.float16)
   b = np.array(b,dtype=np.float16)
   return w , b
def output_case2(z,eta):
  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  sign_z, expo_z , sig_z       = decompose_normalized(z)
  c_z = max(emin-expo_z,0)
   
  sig_eta = np.array(sig_eta,dtype=np.float16)
  sig_z   = np.array(sig_z,dtype=np.float16)
  if sig_eta != 1:
        if - 2 **(1+emin) <= z <0 or (z <  - 2 **(1+emin) and sig_z != 1):
          w = 2 **( -expo_z + expo_eta - c_z )
          b = (-sig_eta + sig_z * 2**(-c_z)) * 2**(expo_eta)
        elif z <  - 2 **(1+emin) and  sig_z == 1:
          w = (1+tmp)*2 **( -expo_z + expo_eta )
          b = (-sig_eta + sig_z + tmp) * 2**(expo_eta)
  else:
        if - 2 **(1+emin) <= z <0 or (z <  - 2 **(1+emin) and sig_z != 1):
           w = 2 **( -1 -expo_z + expo_eta - c_z )
           b = (-1 + sig_z * 2**(-1-c_z)) * 2**(expo_eta)
        else:
           w = 2 **(-expo_z+expo_eta-c_z )
           b = (-1 +  2**(-c_z)) * 2**(expo_eta)
  return w , b
  
def mu_z(z,eta): # Lemma 6
  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  sign_z, expo_z , sig_z       = decompose_normalized(z)
  c_z = max(emin-expo_z,0)

  def func(x):
    if eta >0 and z >=0: # Case 1
      w , b = output_case1(z, eta )
    elif eta <0 and z <0: # Case 2
      w , b = output_case2(z, eta)
    elif eta> 0 and z < 0 :  # Case 3
      nu = - (next_float(eta))
      w_nu , b_nu = output_case2(z, nu)
      w , b       = - w_nu , - b_nu
    else: # Case 4
      nu = - (next_float(eta))
      w_nu , b_nu = output_case1(z, nu )
      w , b       = - w_nu , - b_nu

    w = np.array(w,dtype=np.float16)
    b = np.array(b,dtype=np.float16)

    # print('w,b:', w,b)
    return (x * w) + b
  return func

# Lemma6 verification
f = mu_z(z,eta)
if z>=0 and eta>=0 or (z<0 and eta<0):
  assert f(z) == eta
  assert f(z_plus) == eta_plus
else:
  assert f(z) == eta_plus
  assert f(z_plus) == eta

# f = mu_z(omega,eta)
# assert f(omega) == eta
# assert f(omega_plus) == eta_plus

# f = mu_z(z2,eta)
# assert f(z2) == eta
# assert f(z2_plus) == eta_plus

def lemma15(x,eta):
  n_eta = int( np.array(eta,np.float64)/np.array(omega,np.float64) )
  sign_x, expo_x , sig_x = decompose_normalized(x)
  n_x = int(2**M * sig_x)

  m, r  = n_eta * 2**M // n_x ,  n_eta * 2**M % n_x
  # print(m,r)
  if x <1 and sig_x ==1 :
    y_star = 2.**(-1-M+emin-expo_x)
  elif x<1 and sig_x > 1:
    y_star = 2.**(-M+emin-expo_x)
  elif 1<=x < 1.5 :
    y_star = 2.**(-M+emin)
  y_star = np.array(y_star,dtype=np.float16)
  if r < 2**(M-1) or ( r == 2**(M-1) and n_eta % 2 ==0 ):
    ny1,y2 = m,0
  elif 2**(M-1) < r < 3*2**(M-1) or ( r == 2**(M-1) and n_eta % 2 ==1 ) or ( r == 3* 2**(M-1) and n_eta % 2 ==1 ):
    ny1,y2 = m,y_star
  else:
    ny1,y2 = m+1,0
  y1 = ny1 * np.float_power(2, (-M+emin-expo_x) ).astype(np.float16)
  y1 = np.array(y1,dtype=np.float16)
  y2 = np.array(y2,dtype=np.float16)
  return y1,y2

# Lemma15 verification
y1,y2 = lemma15(z,eta)
assert y1*z+y2*z == eta

K = np.array(1.0,dtype=np.float16)
sig_K_dag = np.array(1.0-tmp,dtype=np.float16)
if eta>0 or (eta<0 and sig_eta != 1):
  expo_zeta = expo_eta - M - 1
else:
  expo_zeta = expo_eta - M - 2
zeta = np.array(2.**(expo_zeta), np.float16)
zeta_minus = before_float(zeta)
ivl_zeta = _Intvl(zeta,zeta)
ivl_zeta_minus = _Intvl(zeta_minus,zeta_minus)


def lemma16(z,eta):
  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  sign_z, expo_z , sig_z       = decompose_normalized(z)
  sign_K, expo_K , sig_K       = decompose_normalized(z)

  m1 = int( np.ceil( (-1+expo_zeta-emin)/ (M+2)  ) )
  beta = [0 for _ in range(m1+1)]
  for i in range(1,m1+1):
    beta[i] = (2-tmp) * np.float_power(2, (expo_zeta-(M+2)*(m1-i)-1))
    beta[i] = np.array(beta[i],dtype=np.float16)
    # alpha_p[i] = sig_K_dag * (2**(expo_zeta-(M+2)*(m1-i)-expo_K))
  lem15_input = np.float_power(2, (expo_zeta-(M+2)*m1))
  lem15_input = np.array(lem15_input,dtype=np.float16) - omega

  tilde_beta1,tilde_beta2 = lemma15(K,lem15_input)
  assert tilde_beta1*K+tilde_beta2*K == lem15_input
  def func(x):
    res = x + tilde_beta1*K+tilde_beta2*K
    for i in range(1,m1+1):
      res = res + beta[i]
    return res
  return func
# verification
f = lemma16(z,eta)
assert f(zero_float) == before_float(zeta)
assert f(omega) == zeta

K = np.array(1.0,dtype=np.float16)
sig_K_dag = np.array(1.0,dtype=np.float16)
if eta>0 or (eta<0 and sig_eta != 1):
  expo_zeta = expo_eta - M - 1
else:
  expo_zeta = expo_eta - M - 2
zeta = np.array(2.**(expo_zeta), np.float16)
sigma_eta = sigma(eta)
sigma_eta_p = sigma(next_float(eta))
sigma_eta_diff =np.abs(sigma_eta_p-sigma_eta)
sign_ed, expo_ed , sig_ed = decompose_normalized(sigma_eta_diff)
e_0 = expo_ed
expo_theta = max(emin-M , -e_0 + emin - M + 1)

def lemma17(z, eta):

    theta = np.array(2.**expo_theta, dtype=np.float16)

    g = lemma16(z,eta)
    beta = np.array(2.**(expo_zeta-M), dtype=np.float16)
    if expo_zeta <=emin or (expo_zeta >=emin+1 and int (sig_eta * 2**M) % 2 == 1 ):
      tilde_beta = 0
    else:
      tilde_beta = beta
    def func(x):
       res = theta * x + (-theta * sigma_eta)
       res = g(res)
      #  print(res + (beta*K))
       res = res + (beta*K) + eta

       return res
    return func
    # print(e_0)
f = lemma17(z,eta)
assert f(sigma(eta)) == eta
assert f(next_float(sigma(eta) )) == next_float(eta)

def lemma18(z,eta):
  L = 1
  tilde_L = L * 2.**(expo_theta+2)
  g = lemma17(z,eta)
  n1 = np.log( ( max_float - sigma(next_float(eta)) ) / ( sigma(eta) - sigma(next_float(eta)) + 2**(expo_zeta-expo_theta)  ) ) / np.log(1/tilde_L)
  n1 = np.maximum(n1, np.log( (sigma(eta) + max_float ) /  2**(expo_zeta-expo_theta)  ) / np.log(1/tilde_L) )
  n1 = int(np.ceil(n1))
  def func(x):
    def g1(x):
      return sigma(g(x))
    res = sigma(x)
    for _ in range(n1+1):
      res = g1(res)
    return res
  return func

# verification
f= lemma18(z,eta)
assert f(eta_plus) == eta_plus
assert f(max_float) == eta_plus
assert f(eta) == eta
assert f(-max_float) == eta

def mu_z_sharp(z,eta): # Lemma 6
  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  sign_z, expo_z , sig_z       = decompose_normalized(z)
  c_z = max(emin-expo_z,0)

  def func(ivl_x):
    if eta >0 and z >=0: # Case 1
      w , b = output_case1(z, eta )
    elif eta <0 and z <0: # Case 2
      w , b = output_case2(z, eta)
    elif eta> 0 and z < 0 :  # Case 3
      nu = - (next_float(eta) )
      w_nu , b_nu = output_case2(z, nu)
      w , b       = - w_nu , - b_nu
    else: # Case 4
      nu =  - (next_float(eta) )
      w_nu , b_nu = output_case1(z, nu )
      w , b       = - w_nu , - b_nu
    # print('w,b:', w,b)
    w = np.array(w,dtype=np.float16)
    b = np.array(b,dtype=np.float16)

    ivl_w = _Intvl(w,w)
    ivl_b = _Intvl(b,b)
    return _intvl_add( _intvl_mul(ivl_w,ivl_x) , ivl_b) # (x * w) + b
  return func

# Lemma6 verification

f = mu_z_sharp(z,eta)
if z>=0 and eta>=0 or (z<0 and eta<0):
  assert f(_Intvl(z,z)) == ivl_eta
  assert f(_Intvl(z_plus,z_plus)) == ivl_eta_plus
else:
  assert f(_Intvl(z,z)) == ivl_eta_plus
  assert f(_Intvl(z_plus,z_plus)) == ivl_eta

# f = mu_z(omega,eta)
# assert f(omega) == eta
# assert f(omega_plus) == eta_plus

# f = mu_z(z2,eta)
# assert f(z2) == eta
# assert f(z2_plus) == eta_plus

K = np.array(1.0,dtype=np.float16)
sig_K_dag = np.array(1.0-tmp,dtype=np.float16)
if eta>0 or (eta<0 and sig_eta != 1):
  expo_zeta = expo_eta - M - 1
else:
  expo_zeta = expo_eta - M - 2
zeta = np.array(2.**(expo_zeta), np.float16)
def lemma16_sharp(eta):
  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  sign_K, expo_K , sig_K       = decompose_normalized(K)

  m1 = int( np.ceil( (-1+expo_zeta-emin)/ (M+2)  ) )
  beta = [0 for _ in range(m1+1)]
  for i in range(1,m1+1):
    beta[i] = (2-tmp) * np.float_power(2, (expo_zeta-(M+2)*(m1-i)-1))
    beta[i] = np.array(beta[i],dtype=np.float16)
    # alpha_p[i] = sig_K_dag * (2**(expo_zeta-(M+2)*(m1-i)-expo_K))
  lem15_input = np.float_power(2, (expo_zeta-(M+2)*m1))
  lem15_input = np.array(lem15_input,dtype=np.float16) - omega

  tilde_beta1,tilde_beta2 = lemma15(K,lem15_input)
  assert tilde_beta1*K+tilde_beta2*K == lem15_input
  def func(itl_x):
    res = _intvl_add(itl_x , _Intvl(tilde_beta1*K,tilde_beta1*K)) #  x + tilde_beta1*K+tilde_beta2*K
    res = _intvl_add(res   , _Intvl(tilde_beta2*K,tilde_beta2*K))
    for i in range(1,m1+1):
      res = _intvl_add(res , _Intvl(beta[i],beta[i])  )#res = res + beta[i]
    return res
  return func
# verification
f = lemma16_sharp(eta)
assert f(ivl_zero) == _Intvl( zeta_minus, zeta_minus )
# assert f(omega) == zeta

K = np.array(1.0,dtype=np.float16)
sig_K_dag = np.array(1.0,dtype=np.float16)
if eta>0 or (eta<0 and sig_eta != 1):
  expo_zeta = expo_eta - M - 1
else:
  expo_zeta = expo_eta - M - 2
zeta = np.array(2.**(expo_zeta), np.float16)
sigma_eta = sigma(eta)
sigma_eta_p = sigma(next_float(eta))
sigma_eta_diff =np.abs(sigma_eta_p-sigma_eta)
sign_ed, expo_ed , sig_ed = decompose_normalized(sigma_eta_diff)
e_0 = expo_ed
expo_theta = max(emin-M , -e_0 + emin - M + 1)

def lemma17_sharp(eta):

    theta = np.array(2.**expo_theta, dtype=np.float16)

    g_sharp = lemma16_sharp(eta)
    beta = np.array(2.**(expo_zeta-M), dtype=np.float16)
    if expo_zeta <=emin or (expo_zeta >=emin+1 and int (sig_eta * 2**M) % 2 == 1 ):
      tilde_beta = 0
    else:
      tilde_beta = beta
    def func(ivl_x):
       ivl_theta = _Intvl(theta,theta)

       res = _intvl_mul(ivl_theta , ivl_x, ) # theta * x + (-theta * sigma_eta)
       res = _intvl_add(res , _Intvl(-theta*sigma_eta, -theta*sigma_eta) )
       res = g_sharp(res)
       res = _intvl_add(res , _Intvl(beta*K,beta*K) ) #  res + (beta*K) + eta
       res = _intvl_add(res , _Intvl(eta,eta))
       return res
    return func
f = lemma17_sharp(eta)
assert f(ivl_sigma_eta) == ivl_eta
# assert f(next_float(sigma(eta) )) == next_float(eta)

def lemma4_sharp(eta):
  L = 1
  tilde_L = L * 2.**(expo_theta+2)
  g_sharp = lemma17_sharp(eta)
  n1 = np.log( ( max_float - sigma(next_float(eta)) ) / ( sigma(eta) - sigma(next_float(eta)) + 2**(expo_zeta-expo_theta)  ) ) / np.log(1/tilde_L)
  n1 = np.maximum(n1, np.log( (sigma(eta) + max_float ) /  2**(expo_zeta-expo_theta)  ) / np.log(1/tilde_L) )
  n1 = int(np.ceil(n1))
  def func(ivl_x):
    def g1_sharp(ivl_x):
      return _intvl_relu(g_sharp(ivl_x))
    res = _intvl_relu(ivl_x)
    for _ in range(n1+1):
      res = g1_sharp(res)
    return res
  return func

# verification
f= lemma4_sharp(eta)
assert f(ivl_eta_plus) == ivl_eta_plus
assert f(ivl_max_float) == ivl_eta_plus
assert f(ivl_eta) == ivl_eta
assert f(_Intvl(-max_float,-max_float)  ) == ivl_eta
assert f(_Intvl(-max_float,max_float)  ) == _Intvl(eta,eta_plus)


def lemma5_sharp(eta, mode=1): 
  # mode 1 : eta -> 0 , eta+ -> 1
  # mode 2 : eta+->1  , eta  -> 0

  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  g_sharp = lemma4_sharp(eta)
  dist = eta_plus - eta 
  dist_inv = 1/dist
  dist_inv = np.array(dist_inv, np.float16)
  if mode == 1:
    def func(ivl_x):
      res = g_sharp(ivl_x)
      res = _intvl_mul(_Intvl(dist_inv,dist_inv) , res) # 2^M x -2^M eta
      res = _intvl_add(res, _Intvl(-dist_inv*eta,-dist_inv*eta) )
      return res
    return func
  elif mode == 2:
    def func(ivl_x):
      res = g_sharp(ivl_x)
      res = _intvl_mul(_Intvl(-dist_inv,-dist_inv) , res) # - 2^M x  + 2^M eta+
      res = _intvl_add(res, _Intvl(dist_inv*eta_plus,dist_inv*eta_plus) )
      return res
    return func

# verification
f= lemma5_sharp(eta)
assert f(ivl_eta) == ivl_zero
assert f(ivl_eta_plus) == ivl_one
assert  f(_Intvl(-max_float,max_float)  ) == _Intvl(zero,one) 
# assert f(_Intvl(-max_float,max_float)  ) == _Intvl(eta,eta_plus)

def lemma2_sharp(z,eta, mode=1): # a,b in [-1,1]
# mode=1 : Indicator i_{>z}
# mode=2 : Indicator i_{<=z}

  sign_eta, expo_eta , sig_eta = decompose_normalized(eta)
  if (z >=0 and eta >=0 ) or (z<0 and eta<0): # z-> eta , z+ -> eta+
    if mode == 1:
      g_sharp = mu_z_sharp(z,eta)
      h_sharp = lemma5_sharp(eta,mode=1)
      def func(ivl_x):
        res = g_sharp(ivl_x)
        res = h_sharp(res)
        return res
      return func
    elif mode ==2:
      g_sharp = mu_z_sharp(z,eta)
      h_sharp = lemma5_sharp(eta,mode=2)
      def func(ivl_x):
        res = g_sharp(ivl_x)
        res = h_sharp(res)
        return res
      return func
  else:
    if mode == 1:
      g_sharp = mu_z_sharp(z,eta)
      h_sharp = lemma5_sharp(eta,mode=2)
      def func(ivl_x):
        res = g_sharp(ivl_x)
        res = h_sharp(res)
        return res
      return func
    elif mode== 2:
      g_sharp = mu_z_sharp(z,eta)
      h_sharp = lemma5_sharp(eta,mode=1)
      def func(ivl_x):
        res = g_sharp(ivl_x)
        res = h_sharp(res)
        return res
      return func
f1 = lemma2_sharp(z,eta,mode=1)  
assert f1(_Intvl(-one,z))  ==  _Intvl(zero,zero)
assert f1(_Intvl(z_plus,one))  ==  _Intvl(one,one)
assert f1(_Intvl(z,z_plus))  ==  _Intvl(zero,one)
assert f1(_Intvl(-one,one))  ==  _Intvl(zero,one)
f2 = lemma2_sharp(z,eta,mode=2)  
assert f2(_Intvl(-one,z))  ==  _Intvl(one,one)
assert f2(_Intvl(z_plus,one))  ==  _Intvl(zero,zero)
assert f2(_Intvl(z,z_plus))  ==  _Intvl(zero,one)
assert f2(_Intvl(-one,one))  ==  _Intvl(zero,one)

def psi_sharp(z,eta, mode=1): # a,b in [-1,1]
# mode=1 : Indicator i_{<z}
# mode=2 : Indicator i_{>=z}
  if mode==1:
    minus_z = np.array(-z,np.float16)
    g_sharp = lemma2_sharp(minus_z,eta, mode=2)
    def func(ivl_x):
      res = _intvl_mul(_Intvl(-one,-one), ivl_x) 
      res = g_sharp(res)
      res = _intvl_add ( _intvl_mul(_Intvl(-one,-one), res) , _Intvl(one,one))
      return res
    return func
  else:
    minus_z = np.array(-z,np.float16)
    g_sharp = lemma2_sharp(minus_z,eta, mode=1)
    def func(ivl_x):
      res = _intvl_mul(_Intvl(-one,-one), ivl_x) 
      res = g_sharp(res)
      res = _intvl_add ( _intvl_mul(_Intvl(-one,-one), res) , _Intvl(one,one))
      return res
    return func

f = psi_sharp(z,eta, mode=1)
assert f(_Intvl(z,one))  ==  _Intvl(zero,zero)
assert f(_Intvl(-one,before_float(z)))  ==  _Intvl(one,one)
assert f(_Intvl(-one,one))  ==  _Intvl(zero,one)

f = psi_sharp(z,eta, mode=2)
assert f(_Intvl(z,one))  ==  _Intvl(one,one)
assert f(_Intvl(-one,before_float(z)))  ==  _Intvl(zero,zero)
assert f(_Intvl(-one,one))  ==  _Intvl(zero,one)

all_fp16 = np.arange(2**16, dtype=np.uint16).view(np.float16)
fp16_in_range = all_fp16[(all_fp16 > -1) & (all_fp16 < 1) & np.isfinite(all_fp16)]

def lemma7(a,b): # [a,b] in [-1,1]
  phi1_sharp = psi_sharp(a,eta, mode=2)
  phi2_sharp = lemma2_sharp(b,eta, mode=2)

  psi1_sharp  = lemma2_sharp(eta,eta,mode=1)  

  alpha = eta- before_float(before_float(eta))
  beta  = before_float(before_float(eta))
  alpha = np.array(alpha,np.float16)
  beta  = np.array(beta,np.float16)

  ivl_alpha = _Intvl(alpha,alpha)
  ivl_beta  = _Intvl(beta,beta)

  def func(ivl_x):
    res = _intvl_add(  _intvl_mul(ivl_alpha ,phi1_sharp(ivl_x) ), _intvl_mul(ivl_alpha ,phi2_sharp(ivl_x) ))
    res = _intvl_add(res,ivl_beta)
    res = psi1_sharp(res)
    return res
  return func





S = [ [zero,zero], [eps,tmp],[one,one] ]

def lemma8(S): # S : collection of intervals
  alpha_pp = np.array(eta_plus - eta,dtype=np.float16)
  psi1_sharp  = lemma2_sharp(eta,eta,mode=1)

  def func(ivl_x):
    res = _Intvl(zero,zero)
    for B in S:
      nu_B = lemma7(B[0],B[1])
      res = _intvl_add(res,  _intvl_mul(  _Intvl(alpha_pp,alpha_pp)  ,  nu_B(ivl_x)) )
    res = _intvl_add(res,  _Intvl(eta,eta))
    res = psi1_sharp(res)
    return res
  return func
  


def target_positive(target):
  def func(x):
    return np.array(np.maximum(target(x),0),np.float16)
  return func
def target_negative(target):
  def func(x):
    return np.array(np.maximum(-target(x),0),np.float16)
  return func
  


    
def lemma9(target):
  target_pos  = target_positive(target)
  target_neg  = target_negative(target)
  
  nu_S_collection_pos = []
  for i, val in enumerate( codomain_range):
    if val ==0:
      continue
    alpha_i = np.array( val-before_float(val) , dtype=np.float16)
    S = []
    a,b = None,None
    for v in domain_range_zero_one: # a b serach 
      if a is None:
        if target_pos(v) >= val:
          a = v
      else:
        if target_pos(v) >= val:
          continue
        else:
          b = before_float(v) 
          a,b = np.array(a,np.float16),np.array(b,np.float16)
          S.append([a,b])
          a,b = None,None

    
    if a is not None:
      b= v
      a,b = np.array(a,np.float16),np.array(b,np.float16)
      S.append([a,b])
    if S:
      for (a,b) in S:
        # print('a b', a,b)
        # print(target_pos(a),target_pos(b),val)
        assert target_pos(a) >= val
        assert target_pos(b) >= val
      pass
      # print("S pos ",S)
    nu_S = lemma8(S)
    nu_S_collection_pos.append([alpha_i,nu_S])



  nu_S_collection_neg = []
  for i, val in enumerate( codomain_range): # negative
    if val ==0:
      continue
    alpha_i = np.array( val-before_float(val) , dtype=np.float16)
    minus_alpha_i =  np.array( -alpha_i , dtype=np.float16)
    S = []
    a,b = None,None
    for v in domain_range_zero_one:
      if a is None:
        if target_neg(v) >= val:
          a = v
      else:
        if target_neg(v) >= val:
          continue
        else:
          b = before_float(v) 
          a,b = np.array(a,np.float16),np.array(b,np.float16)
          S.append([a,b])
          a,b = None,None
    if a is not None:
      b= v
      a,b = np.array(a,np.float16),np.array(b,np.float16)
      S.append([a,b])
    if S:
      for (a,b) in S:
        assert target_neg(a) >= val
        assert target_neg(b) >= val
    nu_S = lemma8(S)
    nu_S_collection_neg.append([minus_alpha_i,nu_S])
   
  
   
  def func(ivl_x):
    nu  = _Intvl(zero,zero)
    for alpha_j,nu_S in nu_S_collection_pos:
      nu = _intvl_add(nu, _intvl_mul(  _Intvl(alpha_j,alpha_j) ,    nu_S(ivl_x)  ))
    for alpha_j,nu_S in nu_S_collection_neg:
      nu = _intvl_add(nu, _intvl_mul(  _Intvl(alpha_j,alpha_j) ,    nu_S(ivl_x)  ))
    return nu
  return func
  
  

def target_sharp(target):
  def func(ivl_x):
    domain_range =  all_fp16[(all_fp16 >= ivl_x.l) & (all_fp16 <= ivl_x.r) & np.isfinite(all_fp16)]
    out = np.array( [target(item) for item in domain_range] , dtype=np.float16 )
    a,b = np.array(out.min(),np.float16), np.array(out.max(),np.float16)
    return _Intvl(a,b)
  return func



# def main():
if True:
    
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument(
        '-v', '--verification_iter', required=False, type=int, default=500)
    parser.add_argument(
        '-t', '--target', required=False, type=str, default='sin')

    args = parser.parse_args()
    target = args.target
    verification_iter = args.verification_iter
    
     
    
    if target == 'sin':
        target = sin
    elif target == 'square':
        target = square
    

    print('building start')
    t= time.time()
    f= lemma9(target)
    print('time for constructing network : ', time.time()-t)
    target_sharp = target_sharp(target)


    for verification_iter in range(verification_iters):
      a,b = np.random.uniform(0.0,1.0,2)
      a,b = min(a,b),max(a,b)
      test_interval = _Intvl( np.array(a,dtype=np.float16), np.array(b,dtype=np.float16) )
      f_sharp_out = f(test_interval)
      print('input interval : {}'.format(test_interval) )
      print('output interval of network : {}'.format(target_sharp(test_interval)))
      print('output interval of target : {}'.format(f_sharp_out))
      assert  f_sharp_out == sin_sharp(test_interval)
      if verification_iter % 10 == 0 :
          print ("{} / {} times verified".format(verification_iter,verification_iters))

    print('verified!')