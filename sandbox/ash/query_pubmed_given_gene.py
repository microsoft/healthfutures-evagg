from Bio import Entrez # Biopython
#from bs4 import BeautifulSoup as BS
import re
import xml.etree.ElementTree as ET

def fetch_details(id_list, email):
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pmc',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

############ Copy to _library.py ############
def search(query, email): # 1
    Entrez.email = email
    handle = Entrez.esearch(db='pmc', 
                            sort='relevance', 
                            retmax='4',
                            retmode='xml', 
                            term=query)
    id_list = Entrez.read(handle)
    return id_list

def find_doi_in_xml(ids): # 2
    handle = Entrez.efetch(db='pmc', id=ids, retmode = 'xml')
    tree = ET.parse(handle)
    root = tree.getroot()
    all_elements = list(root.iter())

    list_dois = []
    for elem in all_elements:
        if((elem.tag == "pub-id") and ("/" in str(elem.text))==True): # doi
             list_dois.append(elem.text)
    return(list_dois, all_elements)   

def find_pmid_in_xml(all_elements): # 3
    list_pmids = []
    for elem in all_elements:
        if((elem.tag == "pub-id") and ("/" not in str(elem.text))==True): # doi
            list_pmids.append(elem.text)
    return(list_pmids)

# for loop across pmids extracted
def get_abstract_and_citation(pmid): # 4
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
    text_info = handle.read()
    citation = generate_citation(text_info)
    abstract = extract_abstract(text_info)
    return(citation, abstract)

def generate_citation(text_info):
    # Extract the author's last name
    match = re.search(r'(\n\n[^\.]*\.)\n\nAuthor information', text_info, re.DOTALL)
    sentence = match.group(1).replace('\n', ' ')
    author_lastname = sentence.split()[0]
    
    # Extract year of publication
    year = re.search(r'\. (\d{4}) ', text_info).group(1)
    
    # Extract journal abbreviation
    journal_abbr = re.search(r'\. ([^\.]*\.)', text_info).group(1).strip(".")
    
    # Extract DOI number
    # TODO: modify to pull from key
    doi_number = re.search(r'\nDOI: (.*)\nPMID', text_info).group(1).replace('10.1111/', '').strip()
    
    # Construct citation
    citation = f"{author_lastname} ({year}), {journal_abbr}., {doi_number}"
    
    return citation

def extract_abstract(text_info):
    # Extract paragraph after "Author information:" sentence and before "DOI:"
    abstract = re.search(r'Author information:.*?\.(.*)DOI:', text_info, re.DOTALL).group(1).strip()    
    return abstract
            
########################

def find_PMIDs_in_text(text):
    pattern = r"StringElement\('(\d+)', attributes=\{'pub-id-type': 'doi'\}\)"
    matches = re.findall(pattern, text)

    # Print all matches (PMIDs)
    for match in matches:
        print("PMID:", match)

def get_pubmed_id(gene_name):
    # Search for the gene name in the 'gene' database
    handle = Entrez.esearch(db='gene', retmax=4, term=gene_name)
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
        
def get_xml_given_pmid(pmid):
    xmls = "https://hanoverdev.blob.core.windows.net/data/20200314_pmc_html/analysis/updates_20210803/20210803/entities/"
    pass

if __name__ == '__main__':
    Entrez.email = "ashleyconard@microsoft.com"
    
    # Get ids for gene name
    email = "ashleyconard@microsoft.com"
    results = search('PRKCG', email)
    id_list = results['IdList']
    print(id_list)
    papers_xml = fetch_details(id_list, email)
    print(papers_xml)
    
    # Get info from ids
    # ids = ['7294636', '8045942', '8857164', '5525338']
    # handle = Entrez.efetch(db='pmc', id=ids, retmode = 'xml')
    # results = handle.read()
    # #print(results)

    # Find PMIDs in papers XML 
    #find_PMIDs_in_text(str(papers_xml))
    #print("hi")
    #abstract = get_abstract('24134140')
    # # Use abstract to determine rare disease papers.
    
    # Get abstract information only
    text_info = get_abstract("22545246")
    print( get_abstract("22545246"))
    print("_____________________________")
    print(extract_abstract(text_info))
        
    # Get XML from hanoverdev given PMID
    #get_xml_given_pmid(37071997)
    
    
    #### Other code no need to read ####
    # Print summary of paper
    # print("summary")
    # summary_handle = Entrez.esummary(db="pubmed", id="19923")
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
    #    print(paper)\