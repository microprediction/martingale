

# A generator for Brownian motion observed with noise, with fluctuating parameters


def brown()->[dict]:
    """

        Generates a list of records {'x': ,'y': } where x is a "true" latent state and y is an observation
        The true latent state x follows a stochastic process as follows:

                dx_t = s_1 exp(v_t) (dW_t + rho dZ_t)
                dv_t = - kappa (v_t) dt + dZ_t + eta dJ_t

        where dJ_t is compensated positive jump process with arrival intensity  gamma, dW_t and dZ_t are dW_t independent Brownian motions;

        The following parameters are generated at the outset:

                s is a scale parameter with extremely diffuse lognormal distribution (almost scale invariant)
                rho is a randomly generated correlation coefficient that is 'almost' diffuse on [-1,1] but puts very little mass on the extremes
                eta is a relative jump size with lognormal distribution
                kappa is a mean reversion parameter in (0.0001,0.1) roughly with diffusion prior

        Furthermore, the y_t observed process is related to x_t in the following manner:

               y_t = x_t + s_1 exp(s_t) epsilon_t       where epsilon_t's are N(0,1) serially correlated with serial correlation nu
                                                            s_t is a slowly varying relative scale parameter (diffuse prior centered at 1)
               ds_t = - tau s_t dt + dX_t       where X_t is Brownian motion  and tau is generated in (0.0001,0.1) roughly with diffusion prior

    :param n:
    :return:   [ {'x': __ ,'y': ___} ]
    """

