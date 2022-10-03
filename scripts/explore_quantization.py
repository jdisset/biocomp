
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{          --     Distance Aware Quantization (modified)     --
# ···············································································


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     Recap of the problem     --
# ···············································································

# - we want to quantize a parameter to a set of values,
#   e.g. transcription rates can only correspond to 2 promoters
# - we want to be able to differentiate through the Quantization
# First solution I used was to simply define a custom jvp for the quantization
# function that would make it look like it was the identity function.
# (I learned later that this is actually a thing lol, it's called STE and has been
# state of the art for a little while)
#
# Other than the fundamental problem of "cheating" the gradient, this solution won't
# work for us because because the thresholds of quantizations themselves are parameters...
# i.e we want to be able to learn that the right quantized rate for hEF1a is a specific value

# So we need to find a way to differentiate through the quantization function while
# still being able to learn the thresholds of the quantization, meaning these threshold
# parameters have to appear in the gradient.

# The intuition is that there must be a way to assign a distance to each quantization thresholds
# and then use that in some kind of exponential weighing scheme to compute the quantized value.

# After a little bit of research, I found this pretty cool 2021 paper:
# https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Distance-Aware_Quantization_ICCV_2021_paper.pdf
# "Distance Aware Quantization"

# It starts from pretty much the same intuition, but they go further and make it really cool, with an
# automatic temperature parameter that makes it act as a perfect quantizer.

# They do not use variable thresholds though, but I think we can adapt their method to our problem.
# Which is what I'm trying to do here, and the main reason why such an assignment-based approach
# is intersting to me.

# I don't think anyone actually cares about variable differentiable quantization values hahaha.
# But I do so let's see if we can make it work.

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     pytorch code from paper     --
# ···············································································
# this is the pytorch version. But I don't want to look at it too much. I need to start from scratch.
def a_soft_argmax(self, x, T, sigma):
    x_floor = x.floor()
    x = x - x_floor.detach()
    m_p = torch.exp(
        -absol.apply(x.unsqueeze(0).repeat(len(self.q_value), 1, 1, 1, 1) - self.q_value)
    )

    # Get the kernel value
    max_value, max_idx = m_p.max(dim=0)
    max_idx = max_idx.unsqueeze(0).float().cuda()
    k_p = torch.exp(-(torch.pow(self.q_value - max_idx, 2).float() / (sigma**2)))

    # Get the score
    score = m_p * k_p

    # Flexible temperature
    denorm = (score[0] - score[1]).abs()
    T_ori = T
    T = T / denorm
    T = T.detach()

    tmp_score = T * score

    # weighted average using the score and temperature

    prob = torch.exp(tmp_score - tmp_score.max())
    denorm2 = prob.sum(dim=0, keepdim=True)
    prob = prob / denorm2

    q_var = self.q_value.clone()
    q_var[0] = q_var[0] - (1 / (torch.exp(torch.tensor(T_ori).float()) - 1))

    q_var[1] = q_var[1] + (1 / (torch.exp(torch.tensor(T_ori).float()) - 1))

    output = (q_var * prob).sum(dim=0)
    output = output + x_floor

    return output


def a_soft_quan(self, x, u, l):
    delta = (u - l) / (self.bit_range)
    interval = (x - l) / delta
    interval = torch.clamp(interval, min=0, max=self.bit_range)
    output = self.a_soft_argmax(interval, a_temperature, a_sigma)
    return output / self.bit_range


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
qvalues = jnp.array([0.0, 0.6, 1.0])
x = 0.7

# compute distance score for each quantization value
dist = jnp.exp(-jnp.abs(jnp.subtract(x, qvalues)))

# let's plot the distance scores for each quantization value
plt.bar(qvalues, dist, width=0.2)
plt.show()
# let's plot a softmax of the distance scores for each quantization values
plt.bar(qvalues, jax.nn.softmax(dist), width=0.2)
plt.show()
# we can accenutate the difference between the two by increasing the a_temperature

# The distance score increases as the normalized input x be-
# comes closer to the quantized value q, and vice versa. 
# he computational cost of computing distance scores
# increases exponentially in accordance with number of possible quantization values.
# In the paper they can use floor and ceil to get the two closest values, 
# but I think we can't do that here because we have a variable number of non-evenly spaced quantization values. 

# Soft assignment:
# We need to assign x to the closest quantization value. Normally we'd use argmax,
# but we need a differentiable version of that. So we use a soft argmax.
# Basically a weighted average of the quantization values, where the weights are distance probabilities.

# distance probabilities are basically a combination of a softmax on the distance scores + a "temperature" parameter (beta),
# aka a scaling factor that makes the probabilities more or less "sharp". God I hate ML jargon.
# Anyway it also uses a 1-d Gaussian kernel (kern_dist) centered on the nearest quantized value
# to make the probabilities more "sharp" around the nearest quantized value.

# but for now let's just use beta, no kernel.
beta = 12
dprob = jax.nn.softmax(beta*dist, axis=0)
plt.bar(qvalues, dprob, width=0.2)
plt.show()

# then we can proceed to the assignment
xq = jnp.dot(qvalues, dprob)

# ok let's summarize in one small func
def soft_quantize(x, qvalues, beta=12):
    dist = jnp.exp(-jnp.abs(jnp.subtract(x, qvalues)))
    dprob = jax.nn.softmax(beta*dist, axis=0)
    return jnp.dot(qvalues, dprob)

# let's plot the soft quantization functions
for beta in [4,8,16,32]:
    x = jnp.linspace(0, 1, 100)
    v = vmap(partial(soft_quantize, qvalues=qvalues, beta=beta))(x)
    plt.plot(x, v, label=r'$\beta$='+str(beta))
plt.legend()
plt.show()

# plot the derivative of the soft quantization functions
# set figsize
plt.figure(figsize=(20, 6))
qvalues = jnp.array([0.0, 0.6, 1.0, 8.0])*0.1
for beta in [16]:
    x = jnp.linspace(0, 1, 10000)
    v = vmap(partial(soft_quantize, qvalues=qvalues, beta=beta))(x)
    dv = vmap(partial(grad(soft_quantize), qvalues=qvalues, beta=beta))(x)
    plt.plot(x, v, label=r'$\beta$='+str(beta))
    # plt.plot(x, dv, label=r'$\beta$='+str(beta)+r' (derivative)')
plt.legend()


# One cool trick they use in the paper is to use a "flexible temperature" parameter that is automatically adjusted.
# Basically you want beta to be super sharp (high) when x is close to a quantization value, and smoother (low) when x is far away.
# we need to estimate the distance between the 2 bounding quantization values
##

qvalues = jnp.array([0.0, 0.6, 1.0])
x = jnp.linspace(0, 1, 100)
# let's plot the distance between the bounding quantization values
# for every x, we find the bounding quantization values and compute the distance between them
bound_len_ground_truth = []
for xx in x:
    a = (qvalues - xx)
    i_upper = jnp.where(a > 0, a, jnp.inf).argmin()
    i_lower = jnp.where(a < 0, a, -jnp.inf).argmax()
    bound = jnp.array([qvalues[i_lower], qvalues[i_upper]])
    bound_len_ground_truth.append(bound[1]-bound[0])
plt.plot(x, bound_len_ground_truth, label='ground truth')
plt.legend()
plt.show()

# # now we need the same thing but differentiable, i.e without argmin and argmax
# # but instead, the soft_argmax
# bound_len = []
# for xx in x:
    # a = (qvalues - xx)
    # i_upper = -jnp.where(a > 0, a, jnp.inf)
    # soft_upper = jax.nn.softmax(5.0*i_upper, axis=0)
    # i_lower = jnp.where(a < 0, a, -jnp.inf)
    # soft_lower = jax.nn.softmax(5.0*i_lower, axis=0)
    # bound = jnp.array([jnp.dot(qvalues, soft_lower), jnp.dot(qvalues, soft_upper)])
    # bound_len.append(bound[1]-bound[0])



# actually, can't we just use the bound_len_ground_truth directly??? does the gradient care? 
# Since we're only using this to compute a dynamic beta...
# let's try it

##

def get_normalized_distance_to_transition(x, qvalues, gamma = 1.0):
    epsilon = 1e-12
    a = (qvalues - x)
    i_upper = jnp.where(a > 0, a, jnp.inf).argmin()
    i_lower = jnp.where(a < 0, a, -jnp.inf).argmax()
    bound = jnp.array([qvalues[i_lower], qvalues[i_upper]])
    bound_len = bound[1]-bound[0]
    # we also need to compute the transition point
    transition_point = (bound[0] + bound[1])/2.0
    return jnp.abs(x-transition_point)/(bound_len + epsilon)
    


# now let's just do a soft quantize where beta is proportional to the ratio of bound_len to the closest quantization value

basebeta = 24.0
qvalues = jnp.array([0.0, 0.6, 1.0])
x = jnp.linspace(0, 1, 1000, endpoint=false)
tr_dists = vmap(partial(get_normalized_distance_to_transition, qvalues=qvalues, gamma=0.001))(x)

betas = jnp.exp(-1000.0*tr_dists)*100.0 + basebeta

constantbeta_v = vmap(partial(soft_quantize, qvalues=qvalues, beta=basebeta))(x)
dynamicbeta_v = vmap(partial(soft_quantize, qvalues=qvalues))(x, beta=betas)

# der_constantbeta_v = vmap(partial(grad(soft_quantize), qvalues=qvalues, beta=basebeta))(x)
# der_dynamicbeta_v = vmap(partial(grad(soft_quantize), qvalues=qvalues))(x, beta=betas)


plt.plot(x, constantbeta_v, label='constant beta')
plt.plot(x, dynamicbeta_v, label='dynamic beta')
# plt.plot(x, der_constantbeta_v, label='der constant beta')
# plt.plot(x, der_dynamicbeta_v, label='der dynamic beta')
plt.legend()
plt.show()

# lmao wait, i don't need this at all........ i will never need to optimize both x and the threshold....




#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
