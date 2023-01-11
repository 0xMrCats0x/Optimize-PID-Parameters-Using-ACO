import utils
import control
import pytest

class TestUtils:
    @staticmethod
    def test_initialize_ants_1():
        ants = utils.initialize_ants(5)
        assert [type(ant) for ant in ants] == [type(utils.Ant()), type(utils.Ant()), type(utils.Ant()), type(utils.Ant()), type(utils.Ant())]

    @staticmethod
    def test_graph_1():
        graph = utils.Graph(3, 5, 5, 5)
        print("k values :")
        print(f"k_p = {graph.k['p']}")
        print(f"k_i = {graph.k['i']}")
        print(f"k_d = {graph.k['d']}")
        print("===============")
        print(f"T values :")
        print(f"T_p = {graph.T['p']}")
        print(f"T_i = {graph.T['i']}")
        print(f"T_d = {graph.T['d']}")
        print("===============")
        print(f"N values :")
        print(f"N_p = {graph.N['p']}")
        print(f"N_i = {graph.N['i']}")
        print(f"N_d = {graph.N['d']}")
        print("===============")
        assert 1 == 0

    @staticmethod
    def test_move_ant_1():
        ant = utils.Ant()
        graph = utils.Graph(3, 5, 5, 5)
        alpha = 0.5
        beta = 0.5
        utils.move_ant(ant, graph, alpha, beta)
        assert ant.path != (0, 0, 0)

    @staticmethod
    def test_calculate_cost_1():
        s = control.TransferFunction.s
        g = 1 / (s * (s + 1) * (s + 5))
        params = {"p": 18, "i": 12.8113879, "d": 6.32232}
        assert pytest.approx(utils.calculate_cost(g, params), abs=25e-1)  == 11.224122

    @staticmethod
    def test_calculate_cost_2():
        s = control.TransferFunction.s
        g = 1 / (s * (s + 1) * (s + 5))
        params = {"p": 39.42, "i": 12.81137972, "d": 30.321864}
        assert pytest.approx(utils.calculate_cost(g, params), abs=25e-1)  == 3.65

    @staticmethod
    def test_update_local_phermone_1():
        s = control.TransferFunction.s
        g = 1 / (s * (s + 1) * (s + 5))
        ant = utils.Ant()
        ant.path = {"i": {"p": 3, "i": 5, "d": 1},
                    "k": {"p": 35.917, "i": 9.9, "d": 33.1579}}
        ant.cost = utils.calculate_cost(g, ant.path["k"])
        graph = utils.Graph(6, 39.42, 12.81137972, 30.321864)
        rho = 0.5
        min_cost = utils.calculate_cost(g, {"p": 39.42, "i": 12.81137972, "d": 30.321864})
        print(min_cost/ant.cost)
        utils.update_local_phermone(ant, graph, min_cost)
        print(graph.T["p"])
        print(graph.T["i"])
        print(graph.T["d"])
        assert 1 == 0


    @staticmethod
    def test_update_global_phermone_1():
        best_ant = utils.Ant(path={"i": {"p": 3, "i": 5, "d": 1}, "k": {}}, cost=11.24)
        graph = utils.Graph(6, 39.42, 12.81137972, 30.321864)
        rho = 0.5
        min_cost = 3.65
        utils.update_global_phermone(best_ant, graph, rho, min_cost)
        print(graph.T["p"])
        print(graph.T["i"])
        print(graph.T["d"])
        assert 1 == 0
