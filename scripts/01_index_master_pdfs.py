from src.indexing.index_build import index_pdfs_master

if __name__ == "__main__":
    stats = index_pdfs_master()
    print(stats)
