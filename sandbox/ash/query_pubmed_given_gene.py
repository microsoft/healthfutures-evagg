from Bio import Entrez # Biopython
from bs4 import BeautifulSoup as BS
import re
import xml.etree.ElementTree as ET

def search(query, email):
    Entrez.email = email
    handle = Entrez.esearch(db='pmc', 
                            sort='relevance', 
                            retmax='4',
                            retmode='xml', 
                            term=query)
    id_list = Entrez.read(handle)
    return id_list

def fetch_details(id_list, email):
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pmc',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results


def get_abstract(pmid):
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
    return handle.read()

def find_pmid_in_xml(xml, ids, all_elements):
    list_pmids = []
    for elem in all_elements:
        if((elem.tag == "pub-id") and ("/" not in str(elem.text))==True): # doi
            list_pmids.append(elem.text)
    return(list_pmids)

def find_doi_in_xml(xml, ids):
    handle = Entrez.efetch(db='pmc', id=ids, retmode = 'xml')
    tree = ET.parse(handle)
    root = tree.getroot()
    all_elements = list(root.iter())

    list_dois = []
    for elem in all_elements:
        if((elem.tag == "pub-id") and ("/" in str(elem.text))==True): # doi
             list_dois.append(elem.text)
    return(list_dois, all_elements)   
            
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

def generate_citation(text_info):
    # Extract the author's last name
    author_lastname = re.search(r'\n\n(\w+)', text_info).group(1)
    
    # Extract the year of publication
    year = re.search(r'\. (\d{4}) ', text_info).group(1)
    
    # Extract the journal abbreviation
    journal_abbr = re.search(r'(\w+)\.', text_info).group(1)
    
    # Extract the DOI number
    doi_number = re.search(r'\nDOI: (.*)\nPMID', text_info).group(1).strip()
    
    # Construct the citation
    citation = f"{author_lastname} ({year}) {journal_abbr}., {doi_number}"
    
    return citation

if __name__ == '__main__':
    # ids = ['7294636', '8045942', '8857164', '5525338']
    # Entrez.email = 'ashleyconard@microsoft.com'
    # handle = Entrez.efetch(db='pmc', id=ids, retmode = 'xml')
    # results = handle.read()
    # #print(results)
    
    # Entrez.email = "ashleyconard@microsoft.com"
    # email = "ashleyconard@microsoft.com"
    # results = search('PRKCG', email)
    # id_list = results['IdList']
    # print(id_list)
    #papers_xml = fetch_details(id_list, email)
    #print(papers_xml)
    
    # Find PMIDs in papers XML 
    #find_PMIDs_in_text(str(papers_xml))
    #print("hi")
    #abstract = get_abstract('24134140')
    # # Use abstract to determine rare disease papers.
    # print(get_abstract("22545246"))
    
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

    citation = get_citation(text_info)
    print(citation)
    citation_eg ="Jezierska (2014) Neurochem., 12491"

    # Test with your specific string
    text_info = "1. J Neurochem. 2014 Mar;128(5):741-51. doi: 10.1111/jnc.12491. Epub 2013 Nov 13.\n\nSCA14 mutation V138E leads to partly unfolded PKCγ associated with an exposed \nC-terminus, altered kinetics, phosphorylation and enhanced insolubilization.\n\nJezierska J(1), Goedhart J, Kampinga HH, Reits EA, Verbeek DS.\n\nAuthor information:\n(1)Department of Genetics, University of Groningen, University Medical Center \nGroningen, Groningen, The Netherlands.\n\nThe protein kinase C γ (PKCγ) undergoes multistep activation and participates in \nvarious cellular processes in Purkinje cells. Perturbations in its \nphosphorylation state, conformation or localization can disrupt kinase \nsignalling, such as in spinocerebellar ataxia type 14 (SCA14) that is caused by \nmissense mutations in PRKCG encoding for PKCγ. We previously showed that SCA14 \nmutations enhance PKCγ membrane translocation upon stimulation owing to an \naltered protein conformation. As the faster translocation did not result in an \nincreased function, we examined how SCA14 mutations induce this altered \nconformation of PKCγ and what the consequences of this conformational change are \non PKCγ life cycle. Here, we show that SCA14-related PKCγ-V138E exhibits an \nexposed C-terminus as shown by fluorescence resonance energy \ntransfer-fluorescence lifetime imaging microscopy in living cells, indicative of \nits partial unfolding. This conformational change was associated with faster \nphorbol 12-myristate 13-acetate-induced translocation and accumulation of fully \nphosphorylated PKCγ in the insoluble fraction, which could be rescued by \ncoexpressing PDK1 kinase that normally triggers PKCγ autophosphorylation. We \npropose that the SCA14 mutation V138E causes unfolding of the C1B domain and \nexposure of the C-terminus of the PKCγ-V138E molecule, resulting in a decrease \nof functional kinase in the soluble fraction. Here, we show that the mutation \nV138E of the protein kinase C γ (PKCγ) C1B domain (PKCγ-V138E), which is \nimplicated in spinocerebellar ataxia type 14, exhibits a partially unfolded \nC-terminus. This leads to unusually fast phorbol 12-myristate 13-acetate-induced \nmembrane translocation and accumulation of phosphorylated PKCγ-V138E in the \ninsoluble fraction, causing loss of the functional kinase. In contrast to \ngeneral chaperones, coexpression of PKCγ's 'natural chaperone', PDK1 kinase, \ncould rescue the PKCγ-V138E phenotype.\n\n© 2013 International Society for Neurochemistry.\n\nDOI: 10.1111/jnc.12491\nPMID: 24134140 [Indexed for MEDLINE] "

