#%%
import pandas as pd
truth = pd.read_csv("data/v1/papers_train_v1.tsv", sep="\t")
output = pd.read_csv(".out/pipeline_benchmark.tsv", sep="\t", header=1)

truth_pmids = set(truth[truth.has_fulltext == True].pmid)
output_pmids = {int(r[1]) for r in output.paper_id.str.split(":")}


fn = truth_pmids - output_pmids
fp = output_pmids - truth_pmids

tp = truth_pmids & output_pmids

precision = len(tp) / (len(tp) + len(fp))
recall = len(tp) / (len(tp) + len(fn))

print(f"Precision: {precision} (N_irrelevant = {len(fp)})")
print(f"Recall: {recall} (N_missed = {len(fn)})")


# %%

from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import CosmosCachingWebClient, get_dotenv_settings

ncbi = NcbiLookupClient(CosmosCachingWebClient(cache_settings=get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_")), settings=get_dotenv_settings(filter_prefix="NCBI_EUTILS_"))
                        
result = {}
pmcids = {}
titles = {}

for pmid in truth_pmids | output_pmids:
    try:
        paper = ncbi.fetch(str(pmid), include_fulltext=True)
        if "pmcid" in paper.props:
            pmcids[pmid] = paper.props["pmcid"]
        titles[pmid] = paper.props["title"]
        if paper is None:
            raise Exception(f"Paper {pmid} not found")
        result[pmid] = "fulltext_xml" in paper.props and paper.props["fulltext_xml"] is not None

    except Exception as e:
        print(f"Error fetching {pmid}: {e}")
        result[pmid] = False

print(any(v==False for v in result.values()))

# %%
