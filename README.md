# Otimização de Mistura de Ração Animal - Algoritmo Simplex

**Disciplina:** Otimização Aplicada
**Professor:** Odemir Moreira da Mata Junior  
**Alunos:** Ana Carla Londero Cazarotto, Emerson Gustavo Pobran Rodrigues, João Vitor Machado Zucchetti e Vanessa da Silva  
**Data:** Novembro/2025

---

## Descrição do Projeto

Este projeto implementa o **Algoritmo Simplex** para resolver um problema real de programação linear: a otimização da formulação de ração para gado de corte, minimizando custos enquanto atende às necessidades nutricionais dos animais.

### Problema

A AgroNutri Ltda. precisa formular uma ração de 1 tonelada que:
- Minimize o custo de produção
- Atenda aos requisitos nutricionais mínimos e máximos
- Respeite as proporções adequadas de cada ingrediente

---

## 🎯 Formulação Matemática

### Variáveis de Decisão
- **x₁** = Milho (kg)
- **x₂** = Farelo de Soja (kg)
- **x₃** = Farelo de Trigo (kg)
- **x₄** = Fosfato Bicálcico (kg)
- **x₅** = Calcário (kg)
- **x₆** = Sal Mineral (kg)

### Função Objetivo
Minimizar:
```
Z = 0,85x₁ + 1,50x₂ + 0,65x₃ + 2,80x₄ + 0,30x₅ + 3,50x₆
```

### Restrições

**Composição total:**
```
x₁ + x₂ + x₃ + x₄ + x₅ + x₆ = 1000
```

**Proteína (140.000g - 180.000g):**
```
140.000 ≤ 90x₁ + 450x₂ + 160x₃ ≤ 180.000
```

**Energia (≥ 2.800.000 kcal):**
```
3350x₁ + 2230x₂ + 1900x₃ ≥ 2.800.000
```

**Cálcio (8.000g - 12.000g):**
```
8.000 ≤ 2x₁ + 3x₂ + 1,5x₃ + 240x₄ + 380x₅ + 120x₆ ≤ 12.000
```

**Fósforo (6.000g - 9.000g):**
```
6.000 ≤ 2,8x₁ + 6,5x₂ + 11x₃ + 185x₄ + 80x₆ ≤ 9.000
```

**Restrições adicionais:**
```
400 ≤ x₁ ≤ 700  (Milho entre 40% e 70%)
x₆ ≤ 10          (Sal Mineral máximo 1%)
xᵢ ≥ 0          (Não-negatividade)
```

---

##  Como Executar o Código

#### Pré-requisitos

- Python 3.7 ou superior
- Biblioteca NumPy

#### Passo 1: Verificar Instalação do Python

Abra o terminal (Linux/Mac) ou Prompt de Comando (Windows) e digite:

```bash
python --version
```

ou

```bash
python3 --version
```

Se aparecer algo como `Python 3.x.x`, você já tem Python instalado!

**Não tem Python?** Baixe em: [python.org/downloads](https://www.python.org/downloads/)

#### Passo 2: Instalar NumPy

No terminal, execute:

```bash
pip install numpy
```

ou

```bash
pip3 install numpy
```

#### Passo 3: Baixar/Copiar o Código

1. Copie o código do arquivo `simplex_racao.py`
2. Salve em um arquivo chamado `simplex_racao.py` no seu computador

#### Passo 4: Executar o Programa

No terminal, navegue até a pasta onde salvou o arquivo e execute:

```bash
python simplex_racao.py
```

ou

```bash
python3 simplex_racao.py
```

---

### **Opção 3: Replit (Online)**

1. Acesse: [replit.com](https://replit.com/)
2. Crie uma conta gratuita
3. Clique em **"+ Create Repl"**
4. Escolha **"Python"** como template
5. Cole o código no editor
6. Clique em **"Run"** 

---

## 🔧 Estrutura do Código

### Classe Principal: `SimplexSolver`

```python
class SimplexSolver:
    """
    Implementação do Algoritmo Simplex para Programação Linear
    Resolve problemas de MINIMIZAÇÃO na forma padrão
    """
```

#### Métodos Principais

| Método | Descrição |
|--------|-----------|
| `__init__()` | Inicializa o problema de PL com função objetivo e restrições |
| `_to_standard_form()` | Converte o problema para forma padrão (adiciona variáveis de folga e artificiais) |
| `_create_initial_tableau()` | Cria o tableau inicial do Simplex (Método Big M) |
| `solve()` | Executa o algoritmo Simplex iterativamente |
| `_is_optimal()` | Verifica se a solução atual é ótima |
| `_select_pivot_column()` | Seleciona a variável de entrada (Regra de Bland) |
| `_select_pivot_row()` | Seleciona a variável de saída (Teste da razão mínima) |
| `_pivot()` | Realiza o pivoteamento no tableau |
| `_extract_solution()` | Extrai a solução ótima do tableau final |

### Função Principal: `resolver_problema_racao()`

Configura e resolve o problema específico da AgroNutri Ltda., apresentando:
- Composição ótima da ração
- Custo total minimizado
- Verificação das restrições nutricionais

---

## Algoritmo Simplex - Detalhes da Implementação

### 1. Conversão para Forma Padrão

O algoritmo converte todas as restrições para a forma padrão:

- **Desigualdades (≤):** Adiciona variáveis de folga
- **Desigualdades (≥):** Multiplica por -1 e adiciona variáveis de folga
- **Igualdades (=):** Adiciona variáveis artificiais (Método Big M)

### 2. Método Big M

Utiliza uma penalidade grande (M = 10⁶) para as variáveis artificiais na função objetivo, forçando-as a sair da base.

### 3. Regra de Bland

Evita ciclagem escolhendo sempre o menor índice em caso de empate.

### 4. Teste da Razão Mínima

Garante que a solução permaneça viável (não-negativa) durante as iterações.

---
