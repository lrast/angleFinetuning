# attempts at a solver for the general system

import numpy as np
import pdb

from scipy.integrate import solve_bvp
from scipy.integrate import quad

from warnings import warn


def defaultInit(p, M, binding, domTheta):
    """ initialization: 
    somewhat nice versions that I made by hand
    """
    ss = np.linspace(p.a, p.b, 300)
    thetaGuess = (domTheta[1] - domTheta[0]) * p.cdf(ss) + domTheta[0]
    qGuess = 5*(thetaGuess - 1.02*domTheta[1])
    yGuess = 5.*M*ss.copy()  # np.zeros(ss.shape) + 0.01

    stateGuess = np.array([thetaGuess, qGuess, yGuess])
    lmbdCGuess = [-1.]

    return ss, stateGuess, lmbdCGuess


def findSol(p, LprimeInv, Cfuns, M, binding=True, domTheta=(0, 1),
            getInitial=defaultInit, lenBinding=True, **solverKwargs):
    """ runs the boundary value problem solver
        M is the value of the constraint, domTheta are theta bounds
    """

    bndConds = getBoundary(binding, lenBinding, M, domTheta)

    # differential equation
    toSolve = lambda t, state, lmbdC: PMPsystem(t, state, lmbdC, p.pdf, LprimeInv, Cfuns)

    # initial conditions
    ss, stateGuess, lmbdCGuess = getInitial(p, M, binding, domTheta)

    # solve it
    sol = solve_bvp(toSolve, bndConds, ss, stateGuess, p=lmbdCGuess, **solverKwargs)
    return sol


def PMPsystem(s, state, lmbdC, pdf, LprimeInv, Cfuns):
    """ System to solve from optimal control.
        lmbdC is the corresponding lagrange multiplier
    """ 
    theta, q, y = state

    C, dC = Cfuns
    lmbdC = lmbdC[0]

    u = LprimeInv(-q / pdf(s))

    dtheta = u
    dq = -lmbdC * dC(theta) * pdf(s)
    dy = C(theta) * pdf(s)

    return np.array([dtheta, dq, dy])


def getBoundary(constrBinding, lenBinding, M, domTheta=(0, 1)):
    """ boundary conditions with the correct slacknesses
    """

    def bndConds(stateI, stateF, lmbdC):
        """ Boundary conditions """
        thetaI = stateI[0]
        thetaF = stateF[0]

        qF = stateF[1]

        yI = stateI[2]  # additional boundary condition: yI starts at 0!
        yF = stateF[2]

        if lenBinding:
            if constrBinding:
                # both binding 
                return np.array([thetaI - domTheta[0], thetaF - domTheta[1],
                                 yI - 0., yF - M])
            else:
                # only the length is binding
                return np.array([thetaI - domTheta[0], thetaF - domTheta[1],
                                yI - 0., lmbdC[0] - 0.])

        else:
            # I'm treating the LHS as binding, but this doesn't have to be the case either
            if constrBinding:
                # only the constraint is binding
                return np.array([thetaI - domTheta[0], qF - 0., yI - 0., yF - M])

            else:
                # neither binding
                return np.array([thetaI - domTheta[0], qF - 0., yI - 0., lmbdC[0] - 0.])

    return bndConds


def ratchetConstraint(p, LprimeInv, Cfuns, M, binding=True, domTheta=(0, 1),
                      maxSteps=30, **solverKwargs):
    """
        Solver that handles non-binding constraints.
        Ratchets down the constraint until it is achieved.
    """
    unConstrained = findSol(p, LprimeInv, Cfuns, M, binding=False, **solverKwargs)

    # could we actually solve the unconstrained version?
    if not unConstrained.success:
        raise Exception('Failure on unconstrained. Bad default init')

    if not binding:
        # unconstrained is what we're looking for
        return unConstrained

    # we are actually constrained
    C = Cfuns[0]
    toInt = lambda s: C(unConstrained.sol(s)[0]) * p.pdf(s)
    M_0 = quad(toInt, p.a, p.b)[0]

    # test if the constraint is slack or tight
    if M > M_0:
        warn('Slack Constraint')
        print('slack', M_0)
        return unConstrained

    memo = {M_0: unConstrained}

    def ratchetDown(M, Mclosest, depth):
        """ Approaches the constrained M from above """
        print(M, Mclosest)
        if M in memo:
            return memo[M]

        if depth > maxSteps:
            raise Exception('Failure to converge to the solution!')

        # initialize to the closest M seen
        testSol = findSol(p, LprimeInv, Cfuns, M, binding=True,
                          getInitial=makeInitializer(memo[Mclosest], Mclosest),
                          **solverKwargs)

        if testSol.success:
            # made it!
            return testSol

        # Else: find the half way point and try again with that point
        Mtest = (M + Mclosest) / 2
        MtestSol = ratchetDown(Mtest, Mclosest, depth+1)
        memo[Mtest] = MtestSol

        return ratchetDown(M, Mtest, depth+1)

    constrainedSol = ratchetDown(M, M_0, 0)
    return constrainedSol


def makeInitializer(previous, Mprev):
    """ for ratcheting down the value of the constraint:
    initialize to previously solved cases """
    def fancyInit(p, M, *args):
        ss = np.linspace(p.a, p.b, 300)
        stateGuess = previous.sol(ss)
        
        stateGuess[0] *= M / Mprev
        stateGuess[1] *= Mprev / M
        stateGuess[2] *= M / Mprev
        
        lmbdCGuess = previous.p
        return ss, stateGuess, lmbdCGuess

    return fancyInit
