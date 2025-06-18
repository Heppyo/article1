#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import pipeline
import pandas as pd
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from transformers import pipeline

# Set your Hugging Face token
from huggingface_hub import login
login("huggingface-toke-here")  # Replace with your actual token

model = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" #Bytt etter nødvendighet

model = pipeline("text-generation", model=MODEL_NAME, device=0)

def create_final_prompt(job_text):
    """Create a meaningful prompt for the LLM to analyze job advertisements."""
    prompt = f"""Du skal analysere denne norske stillingsannonsen og gi strukturert output PÅ NORSK.

KRITISK VIKTIG:
- Svar kun på norsk, selv om annonsen inneholder engelske ord
- Følg NØYAKTIG dette formatet uten ** eller andre markeringer

STILLINGSBESKRIVELSE_START
- List tekniske krav, ansvarsområder og kvalifikasjoner på norsk - en setning per linje med -
STILLINGSBESKRIVELSE_SLUTT

KULTURELLE_SIGNALER_START
- List setninger om organisasjonskultur, verdier og arbeidsmiljø på norsk - en setning per linje med -
KULTURELLE_SIGNALER_SLUTT

YTRE_FORDELER_START
- List setninger om lønn, frynsegoder og karrieremuligheter på norsk - en setning per linje med -
YTRE_FORDELER_SLUTT

IDEELL_KANDIDAT_START
Beskriv hvilke personlige egenskaper, verdier og karaktertrekk arbeidsgiver leter etter - IKKE kvalifikasjoner eller erfaring. Fokuser på personlighet, arbeidsfilosofi og verdier. 2-3 setninger på norsk.
IDEELL_KANDIDAT_SLUTT

ORGANISASJONSVERDIER_START
- List 3-5 hovedverdier som signaliseres på norsk - en verdi per linje med -
ORGANISASJONSVERDIER_SLUTT

ARBEIDSGIVERSTRATEGI_START
Identifiser arbeidsgiverens hovedstrategi for å tiltrekke kandidater. Beskriv strategien med 1-3 ord på norsk som oppsummerer deres tilnærming. Eksempler kan være prestasjonsorientert, familievennlig, innovativ, trygg, samarbeidsorientert, etc. Finn din egen beskrivelse basert på innholdet.
ARBEIDSGIVERSTRATEGI_SLUTT

KULTURELLE_UTTRYKK_START
- List 2-3 språkeksempler som viser kulturelle signaler på norsk - en per linje med -
KULTURELLE_UTTRYKK_SLUTT

OPPSUMMERING_START
Kort oppsummering i 2-3 setninger på norsk om arbeidsgiverens strategi og hvordan de posisjonerer seg for å tiltrekke kandidater
OPPSUMMERING_SLUTT

Stillingsannonse:
{job_text}"""

    return prompt

def analyze_single_job(row):
    job_id = row['stilling_id']
    job_text = row['original_text']

    try:
        prompt = create_final_prompt(job_text)

        # Use Hugging Face model for prediction
        response = model(prompt, max_length=1000)  # Adjust max_length as needed
        analysis = response[0]['generated_text']  # Adjust based on the output structure

        return {
            'stilling_id': job_id,
            'original_text': job_text,
            'analysis': analysis
        }

    except Exception as e:
        logging.error(f"Error analyzing job {job_id}: {e}")
        return {'stilling_id': job_id, 'original_text': job_text, 'error': str(e)}

    except Exception as e:
        logging.error(f"Error analyzing job {job_id}: {e}")
        return {'stilling_id': job_id, 'original_text': job_text, 'error': str(e)}

def run_analysis(input_file, output_file):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting analysis of {input_file}")
    
    df = pd.read_csv(input_file)

    # Begrenser til 5 annonser pga. prosesseringskapasitet
    df = df.head(5)  

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(analyze_single_job, row): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc="Analyzing jobs") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Analysis failed: {e}")
                finally:
                    pbar.update(1)
                    time.sleep(0.3)

    # Lagrer resultater som JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {output_file}")

# Må endre til riktig filplassering    
if __name__ == "__main__":
    run_analysis("C:/Users/can055/article1_training/dataset/cleaned_listings_2020.csv", "job_analysis_results.json")