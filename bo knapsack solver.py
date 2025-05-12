import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import time

st.set_page_config(page_title="Knapsack Pareto Solver", layout="wide")

def knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon):
    start = time.time()
    decision = cp.Variable(len(poids_object), boolean=True)
    contrainte_poid = poids_object @ decision <= poids_maximum
    Z1 = z1_vect @ decision
    p1 = cp.Problem(cp.Maximize(Z1), [contrainte_poid])
    p1.solve(solver=cp.GLPK_MI)

    Z2 = z2_vect @ decision
    p2 = cp.Problem(cp.Maximize(Z2), [contrainte_poid, Z1 == Z1.value])
    p2.solve(solver=cp.GLPK_MI)

    solutions_pareto = [decision.value]
    valeurs_solution_Z1 = [Z1.value]
    valeurs_solution_Z2 = [Z2.value]
    solutions_Z = [(Z1.value, Z2.value)]
    j = 1

    while True:
        p_epsilon = cp.Problem(cp.Maximize(Z1), [contrainte_poid, Z2 >= valeurs_solution_Z2[j - 1] + epsilon])
        p_epsilon.solve(solver=cp.GLPK_MI)
        if Z1.value is None or Z2.value is None:
            break
        solutions_pareto.append(decision.value)
        valeurs_solution_Z1.append(Z1.value)
        valeurs_solution_Z2.append(Z2.value)
        solutions_Z.append((Z1.value, Z2.value))
        j += 1

    end = time.time()

    st.subheader("Solutions efficaces (Pareto):")
    df = pd.DataFrame({
        'Solution': [np.round(sol, 2).tolist() for sol in solutions_pareto],
        'Z1': valeurs_solution_Z1,
        'Z2': valeurs_solution_Z2
    })
    st.dataframe(df, use_container_width=True)

    st.subheader("Front de Pareto")
    fig, ax = plt.subplots()
    ax.plot(valeurs_solution_Z1, valeurs_solution_Z2, 'bo-', linewidth=2)
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_title('Courbe de Pareto')
    st.pyplot(fig)

    st.success(f"Temps de calcul : {round(end - start, 4)} secondes")

def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["⚙️ Saisie Manuelle", "🎲 Génération Aléatoire"],
            default_index=0
        )

    st.title("🧠 Bi-objective Knapsack Solver")
    st.caption("Méthode de résolution : ε-Contrainte | Optimisation multi-objectifs")

    poids_max = st.number_input("Poids maximum du sac", min_value=0.0, step=1.0)
    n = st.number_input("Nombre d'objets", min_value=1, step=1)
    epsilon = st.number_input("Valeur de ε (epsilon)", min_value=0.1, step=0.1)

    poids = np.array([])
    z1 = np.array([])
    z2 = np.array([])

    if selected == "⚙️ Saisie Manuelle":
        st.subheader("Saisie des données des objets")
        for i in range(int(n)):
            cols = st.columns([1, 2, 2, 2])
            with cols[0]:
                st.markdown(f"**Objet {i+1}**")
            with cols[1]:
                p = st.number_input("Poids", key=f"poids_{i}")
            with cols[2]:
                v1 = st.number_input("Valeur Z1", key=f"z1_{i}")
            with cols[3]:
                v2 = st.number_input("Valeur Z2", key=f"z2_{i}")
            poids = np.append(poids, p)
            z1 = np.append(z1, v1)
            z2 = np.append(z2, v2)

        if st.button("Lancer la résolution"):
            knapsack_solver(poids_max, poids, z1, z2, epsilon)

    elif selected == "🎲 Génération Aléatoire":
        st.info("Cliquez sur 'Générer' pour remplir automatiquement les objets.")
        if st.button("Générer"):
            poids = np.random.uniform(1, poids_max, int(n))
            z1 = np.random.randint(10, 100, int(n))
            z2 = np.random.randint(10, 100, int(n))
            df = pd.DataFrame({
                'Objet': [f"Objet {i+1}" for i in range(int(n))],
                'Poids': np.round(poids, 2),
                'Z1': z1,
                'Z2': z2
            })
            st.write(df)
            if st.button("Résoudre avec ces valeurs"):
                knapsack_solver(poids_max, poids, z1, z2, epsilon)

if __name__ == "__main__":
    main()
