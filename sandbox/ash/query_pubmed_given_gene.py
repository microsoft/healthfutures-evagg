from Bio import Entrez # Biopython
from bs4 import BeautifulSoup as BS
import re
#import xml.etree.ElementTree as ET

def get_abstract(pmid):
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
    return handle.read()


def search(query, email):
    Entrez.email = email
    handle = Entrez.esearch(db='pmc', 
                            sort='relevance', 
                            retmax='5',
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

def get_pubmed_id(gene_name):
    # Search for the gene name in the 'gene' database
    handle = Entrez.esearch(db='gene', retmax=5, term=gene_name)
    record = Entrez.read(handle)
    print(record)
    
    # Get the list of IDs from the search results
    id_list = record['IdList']
    print(id_list)
    
    # For each ID in the list, fetch the associated PubMed records
    for gene_id in id_list:
        handle = Entrez.elink(dbfrom="gene", id=gene_id, linkname="gene_pubmed")
        record = Entrez.read(handle)
        print(record)
        
        # Extract and print the PubMed IDs from the linked records
        # for linksetdb in record[0]["LinkSetDb"]:
        #     for link in linksetdb["Link"]:
        #         print("PubMed ID:", link["Id"])

def find_PMIDs_in_text(text):
    pattern = r"StringElement\('(\d+)', attributes=\{'pub-id-type': 'pmid'\}\)"
    matches = re.findall(pattern, text)

    # Print all matches (PMIDs)
    for match in matches:
        print("PMID:", match)
        
def get_xml_given_pmid(pmid):
    xmls = "https://hanoverdev.blob.core.windows.net/data/20200314_pmc_html/analysis/updates_20210803/20210803/entities/"
    pass

if __name__ == '__main__':
    Entrez.email = "ashleyconard@microsoft.com"
    email = "ashleyconard@microsoft.com"
    results = search('SRSF1', email)
    id_list = results['IdList']
    papers_xml = fetch_details(id_list, email)
    #print(papers_xml)
    
    # Find PMIDs in papers XML 
    find_PMIDs_in_text(str(papers))
    
    # Use abstract to determine rare disease papers.
    print(get_abstract("22545246"))
    
    # Get XML from hanoverdev given PMID
    #get_xml_given_pmid(37071997)
    
    
    #### Other code no need to read ####
    # Print summary of paper
    # print("summary")
    # summary_handle = Entrez.esummary(db="structure", id="19923")
    # summary = Entrez.read(summary_handle)
    # summary_handle.close()
    # print(summary)
    
    # Get PubMed ID given gene name
    #get_pubmed_id("SRSF1")
    
    # Beautiful Soup
    # soup = BS(papers, 'xml')
    # D = {}
    # intro = soup.RecordSet.DocumentSummary.Project
    # if intro.ProjectDescr.Name:
    #     D['Name'] = intro.ProjectDescr.Name
        
    #print(D)
    
    #print("papers")
    #print(id_list)
    #print(papers)
    #print(papers[0]['body'])
    #for paper in papers['PubmedArticle']:
    #    print(paper['MedlineCitation']['Article']['ArticleTitle'])
    #    print(paper)
    