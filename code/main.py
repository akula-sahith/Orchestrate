import os
import pandas as pd
from dotenv import load_dotenv
from retriever import get_retriever
from agent import get_agent

# Load environment variables (API Key)
load_dotenv(dotenv_path="../.env")
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def infer_company(issue_text: str) -> str:
    """A lightweight heuristic to infer company if missing."""
    text = issue_text.lower()
    if "claude" in text or "anthropic" in text or "prompt" in text or "model" in text:
        return "Claude"
    elif "hackerrank" in text or "interview" in text or "assessment" in text or "test" in text:
        return "HackerRank"
    elif "visa" in text or "card" in text or "payment" in text or "merchant" in text:
        return "Visa"
    return "Unknown"

def process_tickets(input_csv_path: str, output_csv_path: str):
    print(f"Loading support tickets from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Initialize Vector Store and Agent
    vectorstore = get_retriever(data_dir="../data", persist_directory="./chroma_db")
    agent = get_agent()

    results = []
    
    print(f"\nProcessing {len(df)} tickets...\n" + "-"*40)
    for index, row in df.iterrows():
        issue = str(row.get("issue", row.get("Issue", "")))
        subject = str(row.get("subject", row.get("Subject", "")))
        company = str(row.get("company", row.get("Company", "")))
        
        # 1. Routing / Infer Company
        if pd.isna(row.get("Company", row.get("company"))) or company.strip() == "" or company.lower() in ["none", "nan", "<na>"]:
            inferred = infer_company(issue + " " + subject)
            company = f"{inferred} (Inferred)"
            
        print(f"[{index+1}/{len(df)}] Analyzing ticket for {company}...")
        
        # 2. Filtered Retrieval
        # Map company name to metadata filter
        comp_key = company.lower()
        if "visa" in comp_key:
            filter_val = "visa"
        elif "claude" in comp_key:
            filter_val = "claude"
        elif "hackerrank" in comp_key:
            filter_val = "hackerrank"
        else:
            filter_val = None
            
        search_kwargs = {"k": 6}
        if filter_val:
            search_kwargs["filter"] = {"company": filter_val}
            
        temp_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        search_query = f"{company} support: {subject} {issue}"
        docs = temp_retriever.invoke(search_query)
        context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}" for d in docs])
        
        # 3. Agent Prediction
        try:
            prediction = agent.invoke({
                "company": company,
                "subject": subject,
                "issue": issue,
                "context": context
            })
            
            # Save the parsed prediction
            results.append({
                "status": prediction.status,
                "product_area": prediction.product_area,
                "response": prediction.response,
                "justification": prediction.justification,
                "request_type": prediction.request_type
            })
            print(f"   -> Result: {prediction.status.upper()} ({prediction.request_type})")
        except Exception as e:
            print(f"   -> ERROR processing row {index}: {e}")
            # Fallback for errors to prevent crashing the evaluation
            results.append({
                "status": "escalated",
                "product_area": "Unknown",
                "response": "I apologize, but we are currently experiencing technical difficulties. I am escalating this to a human agent.",
                "justification": f"System error during processing: {str(e)}",
                "request_type": "invalid"
            })
            
    # Write outputs
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print("\n" + "="*40)
    print(f"Finished! Wrote {len(out_df)} predictions to {output_csv_path}")

if __name__ == "__main__":
    # Final production run on the full support tickets dataset
    input_file = "../support_tickets/support_tickets.csv"
    output_file = "../support_tickets/output.csv"
    process_tickets(input_file, output_file)
