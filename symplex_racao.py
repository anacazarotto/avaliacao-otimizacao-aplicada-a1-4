import numpy as np
from typing import Tuple, List
import sys

class SimplexSolver:
    """
    Implementação do Algoritmo Simplex para Programação Linear
    Resolve problemas de MINIMIZAÇÃO na forma padrão
    """
    
    def __init__(self, c, A_eq, b_eq, A_ub, b_ub, bounds):
        """
        Inicializa o problema de PL
        
        Minimizar: c^T * x
        Sujeito a: A_eq * x = b_eq
                   A_ub * x <= b_ub
                   bounds[i][0] <= x[i] <= bounds[i][1]
        """
        self.c_original = np.array(c, dtype=float)
        self.A_eq_original = np.array(A_eq, dtype=float) if len(A_eq) > 0 else None
        self.b_eq_original = np.array(b_eq, dtype=float) if len(b_eq) > 0 else None
        self.A_ub_original = np.array(A_ub, dtype=float) if len(A_ub) > 0 else None
        self.b_ub_original = np.array(b_ub, dtype=float) if len(b_ub) > 0 else None
        self.bounds = bounds
        self.n_vars = len(c)
        
        # Converter para forma padrão
        self._to_standard_form()
        
    def _to_standard_form(self):
        """Converte o problema para forma padrão do Simplex"""
        # Começar com a função objetivo
        c = list(self.c_original)
        
        # Processar restrições de desigualdade (<=)
        A_constraints = []
        b_constraints = []
        n_slack = 0
        
        if self.A_ub_original is not None:
            for i in range(len(self.A_ub_original)):
                row = list(self.A_ub_original[i])
                A_constraints.append(row)
                b_constraints.append(self.b_ub_original[i])
                n_slack += 1
        
        # Processar restrições de igualdade (=)
        n_artificial = 0
        artificial_indices = []
        
        if self.A_eq_original is not None:
            for i in range(len(self.A_eq_original)):
                row = list(self.A_eq_original[i])
                A_constraints.append(row)
                b_constraints.append(self.b_eq_original[i])
                artificial_indices.append(len(A_constraints) - 1)
                n_artificial += 1
        
        # Processar bounds
        for i, (lb, ub) in enumerate(self.bounds):
            # Lower bound: x[i] >= lb => -x[i] <= -lb
            if lb > 0:
                row = [0] * self.n_vars
                row[i] = -1
                A_constraints.append(row)
                b_constraints.append(-lb)
                n_slack += 1
            
            # Upper bound: x[i] <= ub
            if ub < np.inf:
                row = [0] * self.n_vars
                row[i] = 1
                A_constraints.append(row)
                b_constraints.append(ub)
                n_slack += 1
        
        # Criar matriz A completa com variáveis de folga
        n_constraints = len(A_constraints)
        n_total_vars = self.n_vars + n_slack + n_artificial
        
        A = np.zeros((n_constraints, n_total_vars))
        
        # Preencher as variáveis originais
        for i, row in enumerate(A_constraints):
            A[i, :self.n_vars] = row
        
        # Adicionar variáveis de folga
        slack_col = self.n_vars
        for i in range(n_constraints):
            if i not in artificial_indices:
                A[i, slack_col] = 1
                slack_col += 1
        
        # Adicionar variáveis artificiais
        artificial_col = self.n_vars + n_slack
        for idx in artificial_indices:
            A[idx, artificial_col] = 1
            artificial_col += 1
        
        # Função objetivo estendida (Método Big M)
        M = 1e6  # Big M
        c_extended = list(c) + [0] * n_slack + [M] * n_artificial
        
        self.tableau = self._create_initial_tableau(A, b_constraints, c_extended, 
                                                     artificial_indices, n_artificial)
        self.n_vars_total = n_total_vars
        self.n_slack = n_slack
        self.n_artificial = n_artificial
        
    def _create_initial_tableau(self, A, b, c, artificial_indices, n_artificial):
        """Cria o tableau inicial do Simplex"""
        m, n = A.shape
        
        # Tableau: [A | I | b]
        #          [c | 0 | 0]
        tableau = np.zeros((m + 1, n + 1))
        tableau[:m, :n] = A
        tableau[:m, -1] = b
        tableau[-1, :n] = c
        
        # Se há variáveis artificiais, fazer fase I
        if n_artificial > 0:
            for idx in artificial_indices:
                # Subtrair a linha da restrição da linha da função objetivo
                tableau[-1, :] -= tableau[idx, :]
        
        return tableau
    
    def solve(self, max_iterations=1000, tolerance=1e-9):
        """
        Executa o algoritmo Simplex
        
        Returns:
            solution: vetor solução
            optimal_value: valor ótimo da função objetivo
            status: 'optimal', 'unbounded', ou 'infeasible'
        """
        iteration = 0
        
        print("=" * 80)
        print("ALGORITMO SIMPLEX - INÍCIO")
        print("=" * 80)
        print(f"Variáveis originais: {self.n_vars}")
        print(f"Variáveis de folga: {self.n_slack}")
        print(f"Variáveis artificiais: {self.n_artificial}")
        print(f"Total de variáveis: {self.n_vars_total}")
        print()
        
        while iteration < max_iterations:
            # Verificar otimalidade
            if self._is_optimal(tolerance):
                print(f"\n{'=' * 80}")
                print(f"SOLUÇÃO ÓTIMA ENCONTRADA EM {iteration} ITERAÇÕES")
                print(f"{'=' * 80}\n")
                return self._extract_solution()
            
            # Escolher variável de entrada (coluna pivô)
            pivot_col = self._select_pivot_column()
            
            if pivot_col is None:
                return None, None, 'unbounded'
            
            # Escolher variável de saída (linha pivô)
            pivot_row = self._select_pivot_row(pivot_col, tolerance)
            
            if pivot_row is None:
                return None, None, 'unbounded'
            
            print(f"Iteração {iteration + 1}:")
            print(f"  Variável entra: x{pivot_col + 1}")
            print(f"  Variável sai: linha {pivot_row + 1}")
            print(f"  Elemento pivô: {self.tableau[pivot_row, pivot_col]:.4f}")
            
            # Fazer pivoteamento
            self._pivot(pivot_row, pivot_col)
            
            iteration += 1
        
        print("\nNúmero máximo de iterações atingido!")
        return None, None, 'max_iterations'
    
    def _is_optimal(self, tolerance):
        """Verifica se a solução atual é ótima"""
        # Para minimização, ótimo quando todos os custos reduzidos são >= 0
        return np.all(self.tableau[-1, :-1] >= -tolerance)
    
    def _select_pivot_column(self):
        """Seleciona a coluna pivô (regra de Bland - menor índice)"""
        # Encontrar o menor custo reduzido negativo
        costs = self.tableau[-1, :-1]
        negative_costs = np.where(costs < -1e-9)[0]
        
        if len(negative_costs) == 0:
            return None
        
        # Regra de Bland: escolher o menor índice
        return negative_costs[0]
    
    def _select_pivot_row(self, pivot_col, tolerance):
        """Seleciona a linha pivô usando o teste da razão mínima"""
        m = self.tableau.shape[0] - 1
        ratios = []
        
        for i in range(m):
            if self.tableau[i, pivot_col] > tolerance:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append((ratio, i))
        
        if len(ratios) == 0:
            return None
        
        # Escolher a menor razão (regra de Bland em caso de empate)
        ratios.sort()
        return ratios[0][1]
    
    def _pivot(self, pivot_row, pivot_col):
        """Realiza a operação de pivoteamento"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        # Dividir a linha pivô pelo elemento pivô
        self.tableau[pivot_row, :] /= pivot_element
        
        # Eliminar a coluna pivô nas outras linhas
        m = self.tableau.shape[0]
        for i in range(m):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]
    
    def _extract_solution(self):
        """Extrai a solução do tableau final"""
        m, n = self.tableau.shape
        m -= 1  # Remover linha da função objetivo
        n -= 1  # Remover coluna do lado direito
        
        solution = np.zeros(self.n_vars_total)
        
        # Encontrar variáveis básicas
        for j in range(n):
            col = self.tableau[:m, j]
            if np.sum(np.abs(col) > 1e-9) == 1:  # Coluna básica
                i = np.where(np.abs(col) > 1e-9)[0][0]
                if abs(col[i] - 1.0) < 1e-9:
                    solution[j] = self.tableau[i, -1]
        
        # Extrair apenas as variáveis originais
        x = solution[:self.n_vars]
        optimal_value = -self.tableau[-1, -1]  # Negativo para minimização
        
        return x, optimal_value, 'optimal'


def resolver_problema_racao():
    """
    Resolve o problema de otimização da ração da AgroNutri Ltda.
    """
    print("\n" + "=" * 80)
    print("PROBLEMA: OTIMIZAÇÃO DE MISTURA DE RAÇÃO ANIMAL")
    print("EMPRESA: AgroNutri Ltda.")
    print("=" * 80 + "\n")
    
    # Variáveis: x1=Milho, x2=Farelo Soja, x3=Farelo Trigo, 
    #            x4=Fosfato, x5=Calcário, x6=Sal Mineral
    
    # Função objetivo: minimizar custo
    c = [0.85, 1.50, 0.65, 2.80, 0.30, 3.50]
    
    # Restrição de igualdade: soma = 1000 kg
    A_eq = [[1, 1, 1, 1, 1, 1]]
    b_eq = [1000]
    
    # Restrições de desigualdade (<=)
    A_ub = [
        # Proteína máxima: 90x1 + 450x2 + 160x3 <= 180000
        [90, 450, 160, 0, 0, 0],
        
        # Cálcio máximo: 2x1 + 3x2 + 1.5x3 + 240x4 + 380x5 + 120x6 <= 12000
        [2, 3, 1.5, 240, 380, 120],
        
        # Fósforo máximo: 2.8x1 + 6.5x2 + 11x3 + 185x4 + 80x6 <= 9000
        [2.8, 6.5, 11, 185, 0, 80],
        
        # Proteína >= 140000 => -90x1 - 450x2 - 160x3 <= -140000
        [-90, -450, -160, 0, 0, 0],
        
        # Energia >= 2800000 => -3350x1 - 2230x2 - 1900x3 <= -2800000
        [-3350, -2230, -1900, 0, 0, 0],
        
        # Cálcio >= 8000 => -2x1 - 3x2 - 1.5x3 - 240x4 - 380x5 - 120x6 <= -8000
        [-2, -3, -1.5, -240, -380, -120],
        
        # Fósforo >= 6000 => -2.8x1 - 6.5x2 - 11x3 - 185x4 - 80x6 <= -6000
        [-2.8, -6.5, -11, -185, 0, -80],
        
        # Milho <= 700
        [1, 0, 0, 0, 0, 0],
        
        # Sal Mineral <= 10
        [0, 0, 0, 0, 0, 1],
        
        # Milho >= 400 => -x1 <= -400
        [-1, 0, 0, 0, 0, 0],
    ]
    
    b_ub = [180000, 12000, 9000, -140000, -2800000, -8000, -6000, 700, 10, -400]
    
    # Bounds: todas as variáveis >= 0
    bounds = [(0, np.inf) for _ in range(6)]
    
    # Resolver
    solver = SimplexSolver(c, A_eq, b_eq, A_ub, b_ub, bounds)
    solution, optimal_value, status = solver.solve()
    
    if status == 'optimal':
        print("\n" + "=" * 80)
        print("RESULTADOS DA OTIMIZAÇÃO")
        print("=" * 80 + "\n")
        
        ingredientes = ['Milho', 'Farelo de Soja', 'Farelo de Trigo', 
                       'Fosfato Bicálcico', 'Calcário', 'Sal Mineral']
        custos = [0.85, 1.50, 0.65, 2.80, 0.30, 3.50]
        
        print("COMPOSIÇÃO ÓTIMA DA RAÇÃO (por tonelada):")
        print("-" * 80)
        custo_total = 0
        for i, (nome, qtd, custo) in enumerate(zip(ingredientes, solution, custos)):
            custo_ing = qtd * custo
            custo_total += custo_ing
            percentual = (qtd / 1000) * 100
            print(f"{nome:20s}: {qtd:8.2f} kg ({percentual:5.2f}%) - R$ {custo_ing:8.2f}")
        
        print("-" * 80)
        print(f"{'CUSTO TOTAL':20s}: {'':8s} {'':7s}   R$ {custo_total:8.2f}")
        print("=" * 80 + "\n")
        
        # Verificar restrições
        print("COMPOSIÇÃO NUTRICIONAL:")
        print("-" * 80)
        
        proteina = 90*solution[0] + 450*solution[1] + 160*solution[2]
        energia = 3350*solution[0] + 2230*solution[1] + 1900*solution[2]
        calcio = 2*solution[0] + 3*solution[1] + 1.5*solution[2] + 240*solution[3] + 380*solution[4] + 120*solution[5]
        fosforo = 2.8*solution[0] + 6.5*solution[1] + 11*solution[2] + 185*solution[3] + 80*solution[5]
        
        print(f"Proteína: {proteina:,.0f} g (requisito: 140.000 - 180.000 g)")
        print(f"Energia:  {energia:,.0f} kcal (requisito: >= 2.800.000 kcal)")
        print(f"Cálcio:   {calcio:,.0f} g (requisito: 8.000 - 12.000 g)")
        print(f"Fósforo:  {fosforo:,.0f} g (requisito: 6.000 - 9.000 g)")
        print(f"Peso total: {sum(solution):,.0f} kg")
        print("=" * 80 + "\n")
        
    else:
        print(f"\nProblema não resolvido. Status: {status}")


if __name__ == "__main__":
    resolver_problema_racao()