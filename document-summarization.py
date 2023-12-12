import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import csv
import random

join = []
temp = []

with st.sidebar:
  selected = option_menu('Document Summarization', ['Crawling Data', 'Load Data', 'Preprocessing', 'Ekstraksi Fitur', 'Cosinus Similarity', 'Graph', 'Closeness Centrality', 'Pagerank', 'Eignvector Centrality', 'Betweeness Centrality'], default_index=0)
st.title("Document Summarization")
##### Crwaling Data
def crawlingPta():
  st.subheader("Crawling Portal Berita Antaranews")
  url = st.text_input('Inputkan url mediaindonesi berdasarkan topik di sini', 'https://www.antaranews.com/')
  button = st.button('Crawling')
  if (button):
    # masukkan url
    response = requests.get(url) 
    # Isi teks dari respons HTTP yang diterima dari server web setelah melakukan permintaan GET.
    soup = BeautifulSoup(response.text, 'html.parser') 
    # menemukan semua list yang berisi link kategori
    first_page = soup.findAll('li',"dropdown mega-full menu-color1") 

    # menyimpan kategori
    save_categori = []
    for links in first_page:
      categori = links.find('a').get('href')
      save_categori.append(categori)
    # save_categori

    # categori yang akan disearch terdapat pada indeks 1 (politik)
    categori_search = [save_categori[1]] 
    categori_search

    # Inisialisasi list untuk menyimpan data berita
    datas = []

    # Iterasi melalui halaman berita
    for ipages in range(1, 3):

        # Iterasi melalui setiap kategori berita
        for beritas in categori_search:
            # Permintaan untuk halaman berita
            response_berita = requests.get(beritas + "/" + str(ipages))
            namecategori = beritas.split("/")

            # Parsing halaman berita dengan BeautifulSoup
            soup_berita = BeautifulSoup(response_berita.text, 'html.parser')
            pages_berita = soup_berita.findAll('article', {'class': 'simple-post simple-big clearfix'})

            # Iterasi melalui setiap artikel dalam halaman berita
            for items in pages_berita:
                # Mendapatkan link artikel
                get_link_in = items.find("a").get("href")

                # Request untuk halaman artikel
                response_artikel = requests.get(get_link_in)
                soup_artikel = BeautifulSoup(response_artikel.text, 'html.parser')

                # Ekstraksi informasi dari halaman artikel
                judul = soup_artikel.find("h1", "post-title").text if soup_artikel.findAll("h1", "post-title") else ""
                label = namecategori[-1]
                date = soup_artikel.find("span", "article-date").text if soup_artikel.find("span", "article-date") else "Data tanggal tidak ditemukan"

                trash1 = ""
                cek_baca_juga = soup_artikel.findAll("span", "baca-juga")
                if cek_baca_juga:
                    for bacas in cek_baca_juga:
                        text_trash = bacas.text
                        trash1 += text_trash + ' '

                artikels = soup_artikel.find_all('div', {'class': 'post-content clearfix'})
                artikel_content = artikels[0].text if artikels else ""
                artikel = artikel_content.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace(trash1, "").replace("\xa0", "")

                author = soup_artikel.find("p", "text-muted small mt10").text.replace("\t\t", "") if soup_artikel.findAll("p", "text-muted small mt10") else ""

                # Menambahkan data artikel ke dalam list
                datas.append({'Tanggal': date, 'Penulis': author, 'Judul': judul, 'Artikel': artikel, 'Label': label})
    # result = pd.dataFrame(datas)
    st.dataframe(datas)


##### Load Data
def loadData():
  st.subheader("Load Data:")
  data_url = st.text_input('Enter URL of your CSV file here', 'https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_politik_antaranews.csv')

  @st.cache_resource
  def load_data():
      data = pd.read_csv(data_url, index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  # df.set_index('nomor\ufeff', inplace=True)
  # df.index += 1
  df['Artikel'] = df['Artikel'].fillna('').astype(str)
  # if(selected == 'Load Data'):
  st.dataframe(df)
  return (df['Judul'])
    

##### Preprocessing
def preprocessing():
  st.subheader("Preprocessing:")
  st.text("Menghapus karakter spesial")

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_politik_antaranews.csv', index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Artikel'] = df['Artikel'].astype(str).apply(removeSpecialText)
  df['Artikel'] = df['Artikel'].apply(removeSpecialText)
  # df.index += 1
  st.dataframe(df['Artikel'])

  ### hapus tanda baca
  st.text("Menghapus tanda baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Artikel'] = df['Artikel'].apply(removePunctuation)
  st.dataframe(df['Artikel'])

  ### hapus angka pada teks
  st.text("Menghapus angka pada teks")
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Artikel'] = df['Artikel'].apply(removeNumbers)
  st.dataframe(df['Artikel'])

  ### case folding
  st.text("Mengubah semua huruf pada teks menjadi huruf kecil")
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Artikel'] = df['Artikel'].apply(casefolding)
  st.dataframe(df['Artikel'])
  
  #   dfRemoved = pd.DataFrame(removed, columns=['Tokenisasi dan Stopwords']).T
  # # Display the DataFrame
  # st.dataframe(dfRemoved.head(5))


def preprocessingTanpaOutput():

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_politik_antaranews.csv')
      return data

  df = load_data()
  ### if(hapusKarakterSpesial):
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Artikel'] = df['Artikel'].astype(str).apply(removeSpecialText)
  df['Artikel'] = df['Artikel'].apply(removeSpecialText)


  # hapusTandaBaca = st.button("Hapus Tanda Baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Artikel'] = df['Artikel'].apply(removePunctuation)

  ### hapus angka pada teks
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Artikel'] = df['Artikel'].apply(removeNumbers)

  ### case folding
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Artikel'] = df['Artikel'].apply(casefolding)

  return (df["Artikel"], df["Judul"])


##### Ekstraksi Fitur
def ekstraksiFitur():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  st.subheader("Ekstraksi Fitur (TF-IDF):")
  stopwords = stopwords.words('indonesian')

  from sklearn.feature_extraction.text import CountVectorizer

  coun_vect = CountVectorizer(stop_words=stopwords)
  # coun_vect = CountVectorizer()
  count_matrix = coun_vect.fit_transform(preprocessingTanpaOutput()[0])
  count_array = count_matrix.toarray()

  df = pd.DataFrame(data=count_array, columns=coun_vect.vocabulary_.keys())

  # Menampilkan DataFrame menggunakan streamlit
  st.text(ekstraksiFiturTanpaOutput()[0].shape)

  df = pd.concat([preprocessingTanpaOutput()[1], df], axis=1)

  st.dataframe(df)

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  # st.text(tfidf_matrix)
  vsc = pd.DataFrame(data=tfidf_matrix.toarray(),columns = vectorizer.vocabulary_.keys())
  vsc = pd.concat([preprocessingTanpaOutput()[1], vsc], axis=1)
  st.text("Vector Space Model")
  st.dataframe(vsc)


##### Ekstraksi Fitur
def ekstraksiFiturTanpaOutput():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  stopwords = stopwords.words('indonesian')

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  return [tfidf_matrix, tfidf_terms]
  
def cosinusSimilarity():
  import numpy as np

  def buildCosineSimilarity(matrix):
      # Normalisasi vektor
      norm = np.linalg.norm(matrix, axis=1, keepdims=True)
      matrix_norm = matrix / norm

      # Hitung cosine similarity
      calculate_cosine_similarity = np.dot(matrix_norm, matrix_norm.T)

      return calculate_cosine_similarity

  # hitung cosine similarity
  cosine_similarity_result = buildCosineSimilarity(ekstraksiFiturTanpaOutput()[0].toarray())
  # menghasilkan matriks cosine similarity, di mana setiap elemen i, j mewakili cosine similarity antara dokumen i dan j
  cosine_similarity_df = pd.DataFrame(cosine_similarity_result)

  st.subheader("Cosinus Similarity:")
  # Menampilkan DataFrame
  st.dataframe(cosine_similarity_df)
  return cosine_similarity_result

def cosinusSimilarityTanpaOutput():
  import numpy as np

  def buildCosineSimilarity(matrix):
      # Normalisasi vektor
      norm = np.linalg.norm(matrix, axis=1, keepdims=True)
      matrix_norm = matrix / norm

      # Hitung cosine similarity
      calculate_cosine_similarity = np.dot(matrix_norm, matrix_norm.T)

      return calculate_cosine_similarity

  # hitung cosine similarity
  cosine_similarity_result = buildCosineSimilarity(ekstraksiFiturTanpaOutput()[0].toarray())
  # menghasilkan matriks cosine similarity, di mana setiap elemen i, j mewakili cosine similarity antara dokumen i dan j

  # Menampilkan DataFrame
  return cosine_similarity_result

def graph():
  import networkx as nx
  import matplotlib.pyplot as plt

  # Membuat grafik kosong
  G = nx.Graph()

  # Menambahkan node ke dalam grafik
  for i in range(cosinusSimilarityTanpaOutput().shape[0]):
      G.add_node(i)

  # Menambahkan edge berdasarkan cosine similarity
  for i in range(cosinusSimilarityTanpaOutput().shape[0]):
      for j in range(i+1, cosinusSimilarityTanpaOutput().shape[1]):
          # tambahkan threshold jika perlu
          if cosinusSimilarityTanpaOutput()[i, j] > 0.05:
              G.add_edge(i, j, weight=cosinusSimilarityTanpaOutput()[i, j])

  # Menggambar grafik
  fig, ax = plt.subplots()
  nx.draw(G, with_labels=True)
  # plt.show()
  st.subheader("Graph:")
  st.pyplot(fig)

def graphTanpaOutput():
  import networkx as nx

  # Membuat grafik kosong
  G = nx.Graph()

  # Menambahkan node ke dalam grafik
  for i in range(cosinusSimilarityTanpaOutput().shape[0]):
      G.add_node(i)

  # Menambahkan edge berdasarkan cosine similarity
  for i in range(cosinusSimilarityTanpaOutput().shape[0]):
      for j in range(i+1, cosinusSimilarityTanpaOutput().shape[1]):
          # tambahkan threshold jika perlu
          if cosinusSimilarityTanpaOutput()[i, j] > 0.05:
              G.add_edge(i, j, weight=cosinusSimilarityTanpaOutput()[i, j])

  return G

def closenessCentrality():
  import networkx as nx
  # Menghitung closeness centrality
  closeness_centrality = nx.closeness_centrality(graphTanpaOutput())
  st.subheader("Closeness Centrality:")
  # Mencetak hasil
  for node, closeness in closeness_centrality.items():
      st.text(f"Node {node}: Closeness Centrality = {closeness}")

  st.subheader("3 node dengan closeness centrality tertinggi:")
  # Mengurutkan node berdasarkan closeness centrality
  sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
  # Mengambil 3 node dengan closeness centrality tertinggi
  top_3_nodes = sorted_closeness[:3]

  # Membuat DataFrame
  df = pd.DataFrame(top_3_nodes, columns=['Node', 'Closeness Centrality'])

  # Menambahkan isi dari setiap node ke DataFrame
  df['Kalimat'] = [preprocessingTanpaOutput()[0].loc[node] for node, _ in top_3_nodes]
  df.set_index('Node', inplace=True)
  st.dataframe(df)

def pagerank():
  import networkx as nx

  # Menghitung PageRank
  pagerank = nx.pagerank(graphTanpaOutput(), alpha=0.85)
  st.subheader("Pagerank:")
  for node, rank in pagerank.items():
      st.text(f"Node {node}: PageRank = {rank}")

  # Mengurutkan PageRank dari yang tertinggi ke terendah
  sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
  # Mengambil 3 nilai teratas
  st.subheader("3 node dengan nilai pagerank tertinggi:")
  top_3_pagerank = sorted_pagerank[:3]
  df = pd.DataFrame(top_3_pagerank, columns=['Node', 'Pagerank'])

  # Menambahkan isi dari setiap node ke DataFrame
  df['Kalimat'] = [preprocessingTanpaOutput()[0].loc[node] for node, _ in top_3_pagerank]
  # Mengganti indeks DataFrame dengan node
  df.set_index('Node', inplace=True)
  # Menampilkan DataFrame
  st.dataframe(df)

def eignvectorCentrality():
  import networkx as nx
  # Menghitung eigenvector centrality
  eigenvector_centrality = nx.eigenvector_centrality(graphTanpaOutput())
  st.subheader("Eignvector Centrality:")
  # Mencetak hasil
  for node, centrality in eigenvector_centrality.items():
      st.text(f"Node {node}: Eigenvector Centrality = {centrality}")

  # Mengurutkan eigenvector centrality dari yang tertinggi ke terendah
  sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
  # Mengambil 3 nilai teratas
  st.subheader("3 node dengan nilai eignvector centrality tertinggi:")
  top_3_eigenvector = sorted_eigenvector[:3]
  df = pd.DataFrame(top_3_eigenvector, columns=['Node', 'Eigenvector Centrality'])

  # Menambahkan isi dari setiap node ke DataFrame
  df['Kalimat'] = [preprocessingTanpaOutput()[0].loc[node] for node, _ in top_3_eigenvector]
  # Mengganti indeks DataFrame dengan node
  df.set_index('Node', inplace=True)
  # Menampilkan DataFrame
  st.dataframe(df)

def betweennessCentrality():
  import networkx as nx
  # Menghitung betweenness centrality
  betweenness_centrality = nx.betweenness_centrality(graphTanpaOutput())
  st.subheader("Betweenness Centrality:")
  # Mencetak hasil
  for node, centrality in betweenness_centrality.items():
      st.text(f"Node {node}: Betweenness Centrality = {centrality}")

  # Mengurutkan betweenness centrality dari yang tertinggi ke terendah
  sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
  # Mengambil 3 nilai teratas
  st.subheader("3 node dengan nilai betweenness centrality tertinggi:")
  top_3_betweenness = sorted_betweenness[:3]
  df = pd.DataFrame(top_3_betweenness, columns=['Node', 'Betweenness Centrality'])

  # Menambahkan isi dari setiap node ke DataFrame
  df['Kalimat'] = [preprocessingTanpaOutput()[0].loc[node] for node, _ in top_3_betweenness]
  # Mengganti indeks DataFrame dengan node
  df.set_index('Node', inplace=True)
  st.dataframe(df)

  

def main():
  if(selected == 'Crawling Data'):
     crawlingPta()

  if(selected == 'Load Data'):
     loadData()
  if(selected == 'Preprocessing'):
     preprocessing()

  if(selected == 'Ekstraksi Fitur'):
    #  preprocessingOutputHidden()
     ekstraksiFitur()

  if(selected == 'Cosinus Similarity'):
    # st.subheader("Cosinus Similarity:") 
    cosinusSimilarity()
  if(selected == 'Graph'):
     graph()
  if(selected == 'Closeness Centrality'):
     closenessCentrality()
  if(selected == 'Pagerank'):
     pagerank()
  if(selected == 'Eignvector Centrality'):
     eignvectorCentrality()
  if(selected == 'Betweeness Centrality'):
     betweennessCentrality()




if __name__ == "__main__":
    main()

