import math
import numpy as np
import torch
from integral import Simpson, normalpdf

from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#seed_everything(9)

NormalPDF = normalpdf()

#Integral of a normal function cdfd(a,b,u,sigma)
def pdf(x):
  return math.exp(-(x) ** 2 / (2)) / (math.sqrt(2 * math.pi))
 
def sum_fun_xk(xk, func):
  return sum([func(each) for each in xk])
 
def integral(a, b, n, func):
  h = (b - a)/float(n)
  xk = [a + i*h for i in range(1, n)]
  return h/2 * (func(a) + 2 * sum_fun_xk(xk, func) + func(b))
 
def cdfd(a,b,u,o):
  return integral((a-u)/o,(b-u)/o,10000,pdf)

def get_data(m_in,u_in,sigma_in,f_in):

    M = 80 #60-120
    #calculate input and output
    tao = 1
    dt = 0.1 #real phy time speed is um/s
    #m_in = np.random.rand(1)*3 #mN/um in range 3
    #u_in = 4
    #sigma_in = 20
    #f_in = np.random.rand(1)*100 #-100-100uN
    sigma0 = 10
    m_dot_max = 10 #Maximum acceleration
    if sigma_in == 0:
      sigma_in = np.random.rand()
    temp_gamma = cdfd(-sigma0,sigma0,0,sigma_in)/cdfd(-sigma0,sigma0,0,sigma0)

    gamma = min([temp_gamma,1])

    p1 = -M * m_in/(2*(f_in+gamma*u_in))

    k = m_in/(2*p1)

    temp_m_dot = (4*p1*p1-2)*k

    m_dot = min([temp_m_dot,m_dot_max])

    m_out = m_dot * dt + m_in

    u_out = f_in - M * m_out * dt

 
    if np.isnan(m_out) or np.isnan(u_out):
      print("data error")

    return m_out,u_out


def get_data_prefrontal(f_xt,f_yt,u_x,u_y,sigma_x,sigma_y,m_dot_z,belta,lenth,x_sum):

    rca = 0.14
    dt = 0.1
    x_sum += m_dot_z * 10 * dt
    if x_sum < 0:
        x_sum = np.array([0])
    elif x_sum > lenth:
      x_sum = lenth
    flag = np.array([0]) # is it bottom?

    l_at = 2*math.exp(-rca*math.sqrt(f_xt**2+f_yt**2)/(m_dot_z+1e-3)) - 1 #[-1,1]
    belta_next = (1 + l_at) * belta
    belta_next = min(belta_next,1)

    l_pt = np.zeros_like(sigma_x)
    l_pt_temp_x = np.zeros_like(sigma_x)
    l_pt_temp_y = np.zeros_like(sigma_y)
    t_p = np.zeros_like(sigma_y)


    for i in range(len(sigma_x)):
        NormalPDF.change_pram(u=u_x[i],sigma=sigma_x[i])
        l_pt_temp_x[i] = Simpson(NormalPDF.Normal_pdf,-sigma_x[i],sigma_x[i])
        NormalPDF.change_pram(u=u_y[i],sigma=sigma_y[i])
        l_pt_temp_y[i] = Simpson(NormalPDF.Normal_pdf,-sigma_y[i],sigma_y[i])
        NormalPDF.change_pram(u=0,sigma=(i+1)*sigma_x[0])
        t_p_x =  Simpson(NormalPDF.Normal_pdf,-(i+1)*sigma_x[0],(i+1)*sigma_x[0])
        NormalPDF.change_pram(u=0,sigma=(i+1)*sigma_y[0])
        t_p_y = Simpson(NormalPDF.Normal_pdf,-(i+1)*sigma_y[0],(i+1)*sigma_y[0])
        t_p = t_p_x + t_p_y
        l_pt[i] = l_pt_temp_x[i] * l_pt_temp_y[i] / t_p
    l_pt_mean = np.mean(l_pt)

    Tn = 0.5 * l_pt_mean
    m_dot_zt = (1+belta_next+(1+l_at)*(l_pt_mean-Tn))*m_dot_z
    a_hat_zt = (min(m_dot_zt,99.9999)-m_dot_z)/dt

    a_max = 200 #max acc
    if lenth-x_sum < 0:
      a_zt = -a_max
    elif m_dot_zt >= math.sqrt(2*a_max*(lenth-x_sum)):
        a_zt = -a_max
        flag = 1
    elif abs(a_hat_zt) <= a_max:
        a_zt = a_hat_zt
    else:
        a_zt = a_max

    if lenth - x_sum+m_dot_z * dt > 0:
      m_dot_zt_true = max(a_zt * dt + m_dot_z,0)
    else:
      m_dot_zt_true = a_zt * dt + m_dot_z

    return flag, m_dot_zt_true, belta_next, l_pt_mean, x_sum


###########test###############
# m_z = 0
# m_dot_z = 0.1 #[0,80]
# belta = 0.5
# dt = 0.1
# lenth = 2000
# x_sum = 0
# delt_z = 0

# writer = SummaryWriter(comment='controler')

# for i in range(50):
#     f_xt = np.random.rand(1)*1000-500 #[-100,100]
#     f_yt = np.random.rand(1)*1000-500 #[-100,100]

#     #delt_z = (i+1)/500 * 50
#     delt_z = np.array(delt_z)
#     opts = np.array([math.sqrt(delt_z)/4+np.random.randn()*0.1,math.sqrt(delt_z)/3+np.random.randn()*0.1,math.sqrt(delt_z),math.sqrt(delt_z)])
#     opts = np.array([opts*(i+1) for i in range(5)])
#     u_xt = opts[:,0]
#     u_yt = opts[:,1]
#     sigma_xt = opts[:,2]
#     sigma_yt = opts[:,3]

#     #delta_z = np.random.rand(1)*50 #[0,50]
    
    

#     flag, m_dot_zt_true, belta_next, l_pt_mean, x_sum = get_data_prefrontal(f_xt,f_yt,u_xt,u_yt,sigma_xt,sigma_yt,m_dot_z,belta,lenth,x_sum)

#     #print(f_xt,f_yt,m_dot_z,belta,lenth,x_sum)
#     delt_z = delt_z + m_dot_zt_true
#     x_sum = delt_z

#     m_z = np.array(x_sum)
#     m_dot_z = m_dot_zt_true
#     belta = belta_next
#     print("*********************")
#     print(delt_z, m_dot_zt_true)
#     print(flag, m_dot_zt_true, belta_next, l_pt_mean, x_sum)
#     print("*********************")

#     with writer:
#       writer.add_scalar('m_dot_zt_true', m_dot_zt_true, i)
#       writer.add_scalar('flag', flag, i)


