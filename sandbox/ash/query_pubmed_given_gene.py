from Bio import Entrez # Biopython
#from bs4 import BeautifulSoup as BS
import re
import xml.etree.ElementTree as ET
from typing import Set, Dict
from lib.evagg.types import IPaperQuery, Paper, Variant


def fetch_details(id_list, email):
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pmc',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

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

############ Copy to _library.py ############
def search(query): # 1
    id_list = find_ids_for_gene(query)
    return build_papers(id_list)

def find_ids_for_gene(query): # 2
    handle = Entrez.esearch(db='pmc', 
                            sort='relevance', 
                            retmax='1',
                            retmode='xml', 
                            term=query)
    id_list = Entrez.read(handle)['IdList']
    return id_list

def fetch_parse_xml(ids): # 2, previously find_doi_in_xml
    handle = Entrez.efetch(db='pmc', id=ids, retmode = 'xml')
    tree = ET.parse(handle)
    root = tree.getroot()
    all_tree_elements = list(root.iter())

    # list_dois = []
    # for elem in all_elements:
    #     if((elem.tag == "pub-id") and ("/" in str(elem.text))==True): # doi
    #          list_dois.append(elem.text)
    #return(list_dois, all_elements)   
    return(all_tree_elements)

def find_pmid_in_xml(all_tree_elements): # 3
    list_pmids = []
    for elem in all_tree_elements:
        if((elem.tag == "pub-id") and ("/" not in str(elem.text))==True): # pmid
            list_pmids.append(elem.text)
    return(list_pmids)

# for loop across pmids extracted
def get_abstract_and_citation(pmid): # 4
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
    text_info = handle.read()
    citation, doi, pmid = generate_citation(text_info)
    abstract = extract_abstract(text_info)
    return(citation, doi, pmid, abstract)

def generate_citation(text_info):
    # Extract the author's last name
    match = re.search(r'(\n\n[^\.]*\.)\n\nAuthor information', text_info, re.DOTALL)
    sentence = match.group(1).replace('\n', ' ')
    author_lastname = sentence.split()[0]
    
    # Extract year of publication
    year = re.search(r'\. (\d{4}) ', text_info).group(1)
    
    # Extract journal abbreviation
    journal_abbr = re.search(r'\. ([^\.]*\.)', text_info).group(1).strip(".")
    
    # Extract DOI number for citation
    match = re.search(r'\nDOI: (.*)\nPMID', text_info)
    match2 = re.search(r'\nDOI: (.*)\nPMCID', text_info)
    if match:
        doi_number = match.group(1).strip()
    elif match2:
        doi_number = match2.group(1).strip()
    else:
        doi_number = 0.0
    
    # Extract PMID
    match = re.search(r"PMID: (\d+)", text_info)
    if match:
        pmid_number = match.group(1)
    else:
        pmid_number = 0.0
    
    # Construct citation
    citation = f"{author_lastname} ({year}), {journal_abbr}., {doi_number}"
    
    return citation, doi_number, pmid_number

def extract_abstract(text_info):
    # Extract paragraph after "Author information:" sentence and before "DOI:"
    abstract = re.search(r'Author information:.*?\.(.*)DOI:', text_info, re.DOTALL).group(1).strip()    
    return abstract

def build_papers(id_list):
    #papers: List[Paper] = []
    papers_tree = fetch_parse_xml(id_list)
    list_pmids = find_pmid_in_xml(papers_tree)
    
    # Create a dictionary of papers, not part of Papers object
    # papers_dict = {}
    # for pmid in list_pmids:
    #     citation, doi, pmid, abstract = get_abstract_and_citation(pmid)
    #     papers_dict[doi] = {'abstract': abstract, 'citation': citation, 'pmid': pmid}

    # Generate a set of Paper objects
    papers_set = set()
    for pmid in list_pmids:
        citation, doi, pmid, abstract = get_abstract_and_citation(pmid)
        paper = Paper(id=doi, citation=citation, abstract=abstract, pmid=pmid) # make a new Paper object for each entry
        papers_set.add(paper) # add Paper object to set
    
    # for key, value in papers_xml.values():
    #     papers.append(Paper(id=key, **value))
    return papers_set
        
########################

if __name__ == '__main__':
    Entrez.email = "ashleyconard@microsoft.com"
    # result = search('PRKCG')
    # print(result) # of the form DOI: {abstract:___, citation:___, pmid:___}
    
    # Testing Papers code :)
    email = "ashleyconard@microsoft.com"
    id_list = find_ids_for_gene('PRKCG')
    papers_tree = fetch_parse_xml(id_list)
    list_pmids = find_pmid_in_xml(papers_tree)
    print(len(list_pmids))
    count=0
    papers_set = set()
    for pmid in list_pmids:
        count+=1
        print(count, pmid)
        
        citation, doi, pmid, abstract = get_abstract_and_citation(pmid)
        paper = Paper(id=doi, citation=citation, abstract=abstract, pmid=pmid) # make a new Paper object for each entry
        papers_set.add(paper) # add Paper object to set
        if count==3:
            break
    print(papers_set)
    print(type(papers_set))
    #print(papers_dict['10.1038/ncpneuro0289'])
    # papers_xml = fetch_details(id_list, email)
    # print(papers_xml)
    
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
    #handle = Entrez.efetch(db='pubmed', id="22545246", retmode='text', rettype='abstract')
    #text_info = handle.read()
    #print(text_info)
    #print("_____________________________")
    #text_info = get_abstract_and_citation("22545246")
    #citation, pmid, abstract = get_abstract_and_citation("22545246")
    #print(citation)
    #print("_____________________________")
    #print(extract_abstract(text_info))
    
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