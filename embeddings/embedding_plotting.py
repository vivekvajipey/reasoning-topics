import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

def cot_step_print(step_list):
    for i, step in enumerate(step_list):
        print(f"STEP {i}: ", step)

class DimensionalViz:
    def __init__(self, all_embeddings, vis_dimensions=2, model=None, all_sentences=None):
    
        if not all_embeddings and all_sentences and model:
            all_embeddings = model.encode(all_sentences)

        self.all_sentences = all_sentences
        self.all_embeddings = np.vstack(all_embeddings)
        self.vis_dimensions = vis_dimensions
        self.pca_model = PCA(n_components=self.vis_dimensions).fit(self.all_embeddings)

    def understand_pcs(self, pc_num=1, top_k=5):
        projections = self.pca_model.transform(self.all_embeddings)[:, pc_num - 1]
        top_indices = np.argsort(projections)[-top_k:]
        bottom_indices = np.argsort(projections)[:top_k]

        print(f"Sentences with highest projections on PC{pc_num}:")
        for i, idx in enumerate(top_indices):
            print('TOP ', i)
            print(self.all_sentences[idx])

        print(f"\nSentences with lowest projections on PC{pc_num}:")
        for i, idx in enumerate(bottom_indices):
            print('BOTTOM ', i)
            print(self.all_sentences[idx])


    def pca(self, cot_embeddings, cot_strings=None, connections=True, str_labels=False):
        reduced_embeddings = self.pca_model.transform(cot_embeddings)

        indices = np.arange(len(cot_embeddings))
        colormap = plt.cm.viridis
        colors = colormap(indices / max(indices))

        if self.vis_dimensions == 2:
            plt.figure(figsize=(5, 5))
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors)
            if str_labels:
                for i, txt in enumerate(cot_strings):
                    plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
            if connections:
                for i in range(len(reduced_embeddings) - 1):
                    plt.plot(reduced_embeddings[i:i+2, 0], reduced_embeddings[i:i+2, 1], color='grey', alpha=0.5)
            
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('2D Visualization of Embeddings using PCA')
        else:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=colors)
            if str_labels:
                for i, txt in enumerate(cot_strings):
                    ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], txt)
            if connections:
                for i in range(len(reduced_embeddings) - 1):
                    ax.plot(reduced_embeddings[i:i+2, 0], reduced_embeddings[i:i+2, 1], reduced_embeddings[i:i+2, 2], color='grey', alpha=0.5)

            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_zlabel('PCA Component 3')
            ax.set_title('3D Visualization of Embeddings using PCA')

        plt.colorbar(scatter, label='Index in Embeddings Array')
        plt.show()
    
    # def scree_plot(self):
    #     plt.figure(figsize=(5, 5))
    #     explained_variance = np.insert(self.pca_model.explained_variance_ratio_, 0, 0, axis=0)
    #     plt.plot(np.cumsum(explained_variance))
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.title('Scree Plot of PCA')
    #     plt.grid(True)
    #     plt.show()
    
    from sklearn.decomposition import PCA

    def scree_plot(self):
        pca = PCA(n_components=0.99)
        pca.fit(self.all_embeddings)

        plt.figure(figsize=(5, 5))
        explained_variance = np.insert(pca.explained_variance_ratio_, 0, 0, axis=0)
        cumulative_explained_variance = np.cumsum(explained_variance)
        plt.plot(cumulative_explained_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot of PCA')
        plt.grid(True)

        thresholds = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for threshold in thresholds:
            num_components = np.argmax(cumulative_explained_variance >= threshold) + 1
            print(f"{num_components} principal components explain {threshold*100}% of the variance.")

        plt.show()

    def umap(self):
        reducer = umap.UMAP(n_components=self.n_dimensions)
        umap_embeddings = reducer.fit_transform(self.cot_embeddings)
        indices = np.arange(len(umap_embeddings))
        colormap = plt.cm.viridis

        if self.n_dimensions == 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=indices, cmap=colormap, s=15)
            plt.gca().set_aspect('equal', 'datalim')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.title('2D Visualization of Embeddings using UMAP', fontsize=12)
        else:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], c=indices, cmap=colormap, s=15)
            ax.set_xlabel('UMAP Component 1')
            ax.set_ylabel('UMAP Component 2')
            ax.set_zlabel('UMAP Component 3')
            plt.title('3D Visualization of Embeddings using UMAP', fontsize=12)

        plt.colorbar(scatter, label='Index of Embedding')
        plt.show()

    def tsne(self):
        tsne = TSNE(n_components=self.n_dimensions)
        tsne_embeddings = tsne.fit_transform(self.cot_embeddings)
        indices = np.arange(len(tsne_embeddings))
        colormap = plt.cm.viridis

        if self.n_dimensions == 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=indices, cmap=colormap, s=15)
            plt.gca().set_aspect('equal', 'datalim')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('2D Visualization of Embeddings using t-SNE', fontsize=12)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=indices, cmap=colormap, s=15)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            plt.title('3D Visualization of Embeddings using t-SNE', fontsize=12)

        plt.colorbar(scatter, label='Index of Embedding')
        plt.show()


# def plot_with_labels(embed_seq, dim_viz):
#     # number labels
#     labels = [str(i) for i in range(len(embed_seq))]
#     dim_viz.pca(embed_seq[0], cot_strings=labels, connections=True, str_labels=True)