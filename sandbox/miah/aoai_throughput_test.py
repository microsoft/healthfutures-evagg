# %% Imports.

import asyncio
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.evagg.llm import OpenAIClient
from lib.evagg.svc import get_dotenv_settings

# %% Setup.
settings = get_dotenv_settings(filter_prefix="AZURE_OPENAI_")
client = OpenAIClient(settings)

title = """
COG4 mutation in Saul-Wilson syndrome selectively affects secretion of proteins involved in chondrogenesis in chondrocyte-like cells
"""

abstract = """
Saul-Wilson syndrome is a rare skeletal dysplasia caused by a heterozygous mutation in COG4 (p.G516R). Our previous study showed that this mutation affected glycosylation of proteoglycans and disturbed chondrocyte elongation and intercalation in zebrafish embryos expressing the COG4p.G516R variant. How this mutation causes chondrocyte deficiencies remain unsolved. To analyze a disease-relevant cell type, COG4p.G516R variant was generated by CRISPR knock-in technique in the chondrosarcoma cell line SW1353 to study chondrocyte differentiation and protein secretion. COG4p.G516R cells display impaired protein trafficking and altered COG complex size, similar to SWS-derived fibroblasts. Both SW1353 and HEK293T cells carrying COG4p.G516R showed very modest, cell-type dependent changes in N-glycans. Using 3D culture methods, we found that cells carrying the COG4p.G516R variant made smaller spheroids and had increased apoptosis, indicating impaired in vitro chondrogenesis. Adding WT cells or their conditioned medium reduced cell death and increased spheroid sizes of COG4p.G516R mutant cells, suggesting a deficiency in secreted matrix components. Mass spectrometry-based secretome analysis showed selectively impaired protein secretion, including MMP13 and IGFBP7 which are involved in chondrogenesis and osteogenesis. We verified reduced expression of chondrogenic differentiation markers, MMP13 and COL10A1 and delayed response to BMP2 in COG4p.G516R mutant cells. Collectively, our results show that the Saul-Wilson syndrome COG4p.G516R variant selectively affects the secretion of multiple proteins, especially in chondrocyte-like cells which could further cause pleiotropic defects including hampering long bone growth in SWS individuals.
"""

params = {
    "title": title,
    "abstract": abstract,
}


# %% Functions.
async def run_prompt() -> str:
    return await client.prompt_file(user_prompt_file="lib/evagg/content/prompts/paper_finding.txt", params=params)


async def run_prompt_simple() -> str:
    return await client.prompt("Respond with yes or no.")


async def time_prompt() -> float:
    start = time.time()
    await run_prompt()
    return time.time() - start


async def test_value(count: int) -> Tuple[float, List[float]]:
    start = time.time()
    values = await asyncio.gather(*[time_prompt() for _ in range(count)])
    return time.time() - start, values


async def run_test(n: int, concurrency_values: List | None = None, repetitions: int = 10) -> pd.DataFrame:
    if concurrency_values is None:
        concurrency_values = [1, 2, 4, 8, 16, 32, 64]

    df = pd.DataFrame(columns=["concurrency", "iteration", "time", "values"])

    for concurrency_value in concurrency_values:
        for rep in range(repetitions):
            print(f"Testing concurrency value: {concurrency_value}, repetition: {rep + 1}")
            client._config.max_parallel_requests = concurrency_value
            time, values = await test_value(n)
            df.loc[df.shape[0]] = [concurrency_value, rep, time, values]

            print("Cooling down...")
            await asyncio.sleep(60)
            print("...done.")
    return df


async def generate_plot(df: pd.DataFrame, outpath: str, title_str: str) -> None:
    sns.scatterplot(data=df, x="concurrency", y="time")
    median_values = df.groupby("concurrency")["time"].median()

    sns.scatterplot(median_values, marker="D", facecolors="none", edgecolor="orange", s=100)  # type: ignore

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title(title_str)
    plt.savefig(outpath)


# %% Do stuff.


async def main():
    jobs = 64
    df = await run_test(n=jobs)
    df.to_csv(".out/throughput_test.csv")
    await generate_plot(df, ".out/throughput_test.png", f"Throughput (jobs={jobs})")


if __name__ == "__main__":
    asyncio.run(main())

# %%