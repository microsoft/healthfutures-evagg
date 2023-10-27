from Bio import Entrez # Biopython

def search(query, email): # 
    Entrez.email = email
    handle = Entrez.esearch(db='pmc', 
                            sort='relevance', 
                            retmax='20',
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list, email):
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pmc',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

if __name__ == '__main__':
    email = "ashleyconard@microsoft.com"
    results = search('SRSF1', email)
    id_list = results['IdList']
    papers = fetch_details(id_list, email)
    for paper in papers['PubmedArticle']:
        print(paper['MedlineCitation']['Article']['ArticleTitle'])