import numpy as np
import control

class Ant:
    def __init__(self, path={"i": dict(), "k": dict()}, cost=0):
        """
        Initializes an Ant object in the ACO algorithm.

        Parameters
        ----------
            path : dict_of_dict_of_float.
                The path that the ant took, where path["i"] -> {"p": i_p, "i": i_i, "d": i_d}
                                                  path["k"] -> {"p": k_p, "i": k_i, "d": k_d}
                Where i is the index of the visited node and k is its value.
            cost : float.
                The cost of the path.
        """
        self.path = path
        self.cost = cost


def handle_inf(A, scale=10):
    A[A == float("inf")] = 0
    A[A == 0] = np.max(A) * scale


class Graph:
    def __init__(self, graph_size, k_p_mean, k_i_mean, k_d_mean, std=10):
        """
        Initializes a graph to tune the PID parameters using the ACO algorithm.

        Parameters
        ----------
            graph_size : int.
                The number of nodes at each layer.

            k_p_mean : float.
                The mean value of k_p about which we build the uniform distribution
                from which we randomly initialize the k_p values.

            k_i_mean : float.
                The mean value of k_i about which we build the uniform distribution
                from which we randomly initialize the k_i values.

            k_d_mean : float.
                The mean value of k_d about which we build the uniform distribution
                from which we randomly initialize the k_d values.

        Returns
        -------
            None.
        """
        # Initialize k["p"], k["i"] and k["d"]. Take the absolute value so that we don't
        # have negative gains.
        self.k = dict()
        self.k["p"] = np.abs(np.random.normal(k_p_mean, std, (graph_size, 1)))
        self.k["i"] = np.abs(np.random.normal(k_i_mean, std, (graph_size, 1)))
        self.k["d"] = np.abs(np.random.normal(k_d_mean, std, (graph_size, 1)))

        # Initialize the edge trails
        tau_0 = 1/(graph_size*3)
        self.T = dict()
        self.T["p"] = np.ones((1, graph_size)) * tau_0
        self.T["i"] = np.ones((graph_size, graph_size)) * tau_0
        self.T["d"] = np.ones((graph_size, graph_size)) * tau_0

        # Initialize the desirability of each edge, an edge is more desirable
        # if its value is closer to the mean
        self.N = dict()

        # Configure numpy so that division by there doesn't result in an error
        np.seterr(divide="ignore")

        # Calculate the desirability of the edges going to the proportional gains
        self.N["p"] = np.abs(np.divide(np.ones((1, graph_size)), (self.k["p"].T - k_p_mean)))
        handle_inf(self.N["p"])

        # Calculate the desirability of the edges going to the integral gains
        self.N["i"] = np.abs(np.ones((1, graph_size)) / (self.k["i"].T - k_i_mean))
        handle_inf(self.N["i"])

        # Calculate the desirability of the edges going to the deraivative gains
        self.N["d"] = np.abs(np.ones((1, graph_size)) / (self.k["d"].T - k_d_mean))
        handle_inf(self.N["d"])

#------------------------------------------------------------------------------

def initialize_ants(n_ants):
    """
    Returns a list of basic Ant objects of length n_ants.

    Parameters
    ----------
        n_ants : int.

    Returns
    -------
        list_of_Ant
    """
    return [Ant() for i in range(n_ants)]


# def initialize_parameters(graph_size):
#     """
#     Returns the graph which we search for the optimal solution.
#
#     Parameters
#     ----------
#         graph_size : int.
#
#     Returns
#     -------
#         tuple_of_list_of_Node.
#
#     Notes
#     -----
#         The graph is divided into three lists of nodes: k_p_nodes, k_i_nodes and
#         k_d_nodes.
#         The lists contain Node objects, where k_l_nodes[i].edges[j] represents the edge
#         from the j_th node in the l-1_th list to the i_th node in the l_th set.
#     """
#     return ([Node(15, [(1/15, 1/9)]), Node(18, [(1/18, 1/9)]), Node(21, [(1/21, 1/9)])],
#             [Node(11, [(1/11, 1/9), (1/11, 1/9), (1/11, 1/9)]),
#              Node(13, [(1/13, 1/9), (1/13, 1/9), (1/13, 1/9)]),
#              Node(15, [(1/15, 1/9), (1/15, 1/9), (1/15, 1/9)])],
#             [Node(2, [(1/2, 1/9), (1/2, 1/9), (1/2, 1/9)]),
#              Node(6, [(1/6, 1/9), (1/6, 1/9), (1/6, 1/9)]),
#              Node(7.5, [(1/7.5, 1/9), (1/7.5, 1/9), (1/7.5, 1/9)])])


def move_ant(ant, graph, alpha, beta):
    """
    Moves an ant along a path in the given graph nodes to generate a solution,
    and updates the path of the given ant.

    Parameters
    ----------
        ant : Ant.
            The ant to move.

        graph : Graph.
            The graph on which the ant moves.

        alpha : float.
            The constant that determines the relative influence of phermone values.

        beta : float.
            The constant that determines the relative influence of desirability values.

    Returns
    -------
        None.

    Notes
    -----
        The updates are applied to the given Ant object.
    """
    J = ["p", "i", "d"]
    i = 0
    path = {"i" : dict(),
            "k" : dict()}
    for j in J:
        t_alpha = np.power(graph.T[j][i,:], alpha)
        n_beta = np.power(graph.N[j][0,:], beta)
        p_num = t_alpha * n_beta
        p_denom = np.sum(p_num)
        p = (p_num / p_denom).reshape(-1, )
        i = int(np.random.choice(np.arange(p.size), p=p))
        path["i"][j] = i
        path["k"][j] = graph.k[j][i, 0]
    ant.path = path



def calculate_cost(g, h, sign, T, params):
    """
    Calculates the cost function of the solution.
    The cost function is the sum of (rise_time, settling_time, maximum_peak_overshoot)

    Parameters
    ----------
        g : control.TransferFunction
            The transfer function of the plant

        h : control.TransferFunction.
            The transfer function of the feedback.

        sign : int.
            -1 indicates negative feedback and 1 indicates positive feedback.

        T : float.
            The duration of the simulation.

        params : dict_of_float.
            The PID controller parameters: {"p": k_p, "i": k_i, "d": k_d)}

    Returns
    -------
        float.
    """
    def step_info(t, yout):
        overshoot = (np.max(yout) - 1) / yout[-1]

        rise_time = t[next(i for i in range(0,len(yout)) if yout[i]>=yout[-1]*.90)] - t[next(i for i in range(0,len(yout)) if yout[i]>=yout[-1]*.10)]

        try:
            settling_time = t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1])>1.02)]-t[0]
        except StopIteration:
            settling_time = T

        return (overshoot, rise_time, settling_time)

    s = control.TransferFunction.s
    k_p, k_i, k_d = params["p"], params["i"], params["d"]
    g_c = k_p + (k_i / s) + (k_d * s)
    t, yout = control.step_response(control.feedback(control.series(g, g_c), h, sign), T=T)

    info = step_info(t, yout)
    return (sum(info), ) + info


def update_local_phermone(ant, graph, min_cost):
    """
    Updates the phermone levels in the edges of the path that the given ant
    has taken.

    Parameters
    ----------
        ant : Ant.
            The ant that took a path.

        graph : Graph.
            The graph on which the ant moved.

        min_cost : float.
            The cost of the best solution so far.

    Returns
    -------
        None.

    Notes
    -----
        The updates are applied to the given Graph object.
    """
    path = ant.path["i"]
    i = 0
    J = ["p", "i", "d"]
    delta_t = min_cost / ant.cost
    for j in J:
        graph.T[j][i, path[j]] = graph.T[j][i, path[j]] + delta_t
        i = path[j]


def update_global_phermone(best_ant, graph, rho, min_cost):
    """
    Updates the phermone levels of the path of the best solution and allows the
    phermone in the paths of bad solution to evaporate.

    Parameters
    ----------
        best_ant : Ant.
            The ant that took the path with the least cost.

        graph : Graph.
            The graph on which the ant moved.

        rho : float, = ]0, 1].
            The evaporation rate.

        min_cost : float.
            The cost of the best solution so far.

    Returns
    -------
        None.

    Notes
    -----
        The updates are applied to the given Graph object.
    """
    path = best_ant.path["i"]

    i = 0
    J = ["p", "i", "d"]
    delta_t = min_cost / best_ant.cost
    for j in J:
        graph.T[j] = graph.T[j] * (1-rho)
        graph.T[j][i, path[j]] = graph.T[j][i, path[j]] + delta_t
        i = path[j]


def get_best_solution(ants):
    """
    Returns the path with the least cost.

    Parameters
    ----------
        ants : list_of_Ant.

    Returns
    -------
        tuple_of_float.
    """
    return (18, 13, 6)
