import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from RNA import fold

def circle_plot(structure, sequence):
    n = len(sequence)
    
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(8, 8))  
    ax.set_aspect('equal')


    for i in range(n):
        ax.plot(x[i], y[i], 'ro', markersize=4) 


    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i) 
        elif char == ')':
            if stack:
                j = stack.pop() 
                ax.plot([x[j], x[i]], [y[j], y[i]], 'b-', linewidth=0.5)  
                ax.plot(x[i], y[i], 'bo', markersize=4) 
                ax.plot(x[j], y[j], 'bo', markersize=4)  
            else:
                print(f"Warning: Closing bracket ')' at index {i} has no matching opening bracket.")

    if stack:
        print(f"Warning: Opening bracket '(' at indices {stack} has no matching closing bracket.")

    quarter_indices = [i * (n // 4) for i in range(4)]  
    for i in quarter_indices:
        label_x = 1.05 * x[i]  
        label_y = 1.05 * y[i]
        ax.text(label_x, label_y, str(i + 1), fontsize=8, ha='center', va='center', 
                color='black', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Unpaired', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Paired', markerfacecolor='blue', markersize=8)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))

    ax.set_axis_off()
    plt.title("RNA Secondary Structure - Circle Plot")
    st.pyplot(fig)

def mountain_plot(sequence, structure):
    n = len(sequence)
    base_pairs = [0] * n
    stack = []

def count_base_pairs(structure):
    """
    Count the number of enclosing base pairs for each position in the RNA sequence.
    """
    n = len(structure)
    base_pair_counts = np.zeros(n, dtype=int)
    stack = []

    for i in range(n):
        if structure[i] == '(':
            stack.append(i) 
        elif structure[i] == ')':
            if stack:
                j = stack.pop()  
                base_pair_counts[i] += 1
                base_pair_counts[j] += 1
                
                for k in range(j + 1, i):
                    base_pair_counts[k] += 1

    return base_pair_counts

def mountain_plot(base_pair_counts, sequence):
    """
    Generate a mountain plot based on the number of enclosing base pairs.
    """
    n = len(sequence)
    x = np.arange(n)

    plt.figure(figsize=(12, 6))

    colors = {'A': 'cyan', 'C': 'blue', 'G': 'yellow', 'U': 'red'}
    legend_labels = {nucleotide: color for nucleotide, color in colors.items()}

    for i, base in enumerate(sequence):
        color = colors.get(base, 'black')  # Default color for unexpected bases
        plt.plot(x[i], base_pair_counts[i], 'o', color=color, markersize=5)

    plt.fill_between(x, base_pair_counts, color='gray', alpha=0.5)
    plt.title("Mountain Graph of RNA Sequence")
    plt.xlabel("Sequence Position")
    plt.ylabel("Number of Enclosing Base Pairs")
    plt.xticks(ticks=x[::4]) 
    plt.grid()

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
    plt.legend(handles, legend_labels.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))
    
    st.pyplot(plt)

def count_base_pairs(structure):
    """
    Count the number of enclosing base pairs for each position in the RNA sequence.
    """
    n = len(structure)
    base_pair_counts = np.zeros(n, dtype=int)
    stack = []

    for i in range(n):
        if structure[i] == '(':
            stack.append(i)
        elif structure[i] == ')':
            if stack:
                j = stack.pop()
                base_pair_counts[i] += 1
                base_pair_counts[j] += 1
                for k in range(j + 1, i):
                    base_pair_counts[k] += 1

    return base_pair_counts

def secondary_structure_plot(sequence, structure):
    n = len(sequence)
    fig, ax = plt.subplots(figsize=(15, 6))

    x_vals = np.arange(1, n + 1)
    y_vals = np.zeros(n)  
    dot_colors = ['red'] * n  

    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()

                pair_type = f"{sequence[i]}{sequence[j]}"
                if pair_type in ['AU', 'UA']:
                    dot_colors[i] = dot_colors[j] = 'cyan'
                elif pair_type in ['GU', 'UG']:
                    dot_colors[i] = dot_colors[j] = 'yellow'
                elif pair_type in ['GC', 'CG']:
                    dot_colors[i] = dot_colors[j] = 'blue'

                arch_height = (j - i) / 2.0

                arch_x = np.linspace(x_vals[i], x_vals[j], 100)
                arch_y = arch_height * np.sqrt(1 - ((arch_x - (x_vals[i] + x_vals[j]) / 2) / (j - i) * 2) ** 2) * -1

                ax.plot(arch_x, arch_y, color='black', linewidth=0.8)

    ax.scatter(x_vals, y_vals, c=dot_colors, s=100)

    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Relative Distance Between Paired Bases")
    ax.set_title("RNA Secondary Structure - Base Pairing Plot")

    legend_labels = {
        'cyan': 'AU/UA pairs',
        'yellow': 'GU/UG pairs',
        'blue': 'GC/CG pairs',
        'red': 'Unpaired'
    }
    for color, label in legend_labels.items():
        ax.scatter([], [], c=color, label=label)
    
    plt.legend(loc='upper right')
    st.pyplot(fig)


def fold_rna(sequence):
    structure, _ = fold(sequence)
    return structure

st.title('RNA Sequence Secondary Structure Analysis')
with st.sidebar:
    st.title("sTARA 🧬")
    st.write("Making information on space biology understandable and accessible to all!")

    st.write("# Contents")
    st.write("## Main: The Homepage")
    st.write("## Basic Information: Speak With a Chatbot")
    st.write("## Compare and Contrast: Compare Observations Between Studies")
    st.write("## Sequencing: Make Observations of and Comparisons Between Raw Sequence Files")
    st.write("## Quality Scores: Evaluate the Qualities of Gene Samples")
    st.write("## Structure: Observing the RNA Sequence Individual Structure in Various Graphs")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'SeqID' in df.columns and 'sequence' in df.columns and 'quality' in df.columns:
        seq_id = st.selectbox("Choose a Sequence ID", df['SeqID'].unique())        
        selected_seq = df[df['SeqID'] == seq_id]['sequence'].values[0]
        selected_seq = selected_seq.replace('T', 'U')

        rna_structure = fold_rna(selected_seq)

        graph_type = st.selectbox("Choose a Graph Type", ["Circle Plot", "Mountain Plot", "Secondary Structure Plot"])

        if graph_type == "Circle Plot":
            circle_plot(rna_structure, selected_seq)
        elif graph_type == "Mountain Plot":
            base_pair_counts = count_base_pairs(rna_structure)
            mountain_plot(base_pair_counts, selected_seq)   
        elif graph_type == "Secondary Structure Plot":
            secondary_structure_plot(selected_seq, rna_structure)
    
        
    else:
        st.error("CSV file doesn't have the required columns: SeqID, sequence, quality.")
