import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load candidates data from CSV
def fetch_candidates(f_name):
    cand_list = []
    with open(f_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cand_list.append({
                "id": int(row['id']),
                "job_desc_exp": row['job_desc_exp'],
                "location": row['location'],
                "salary": int(row['salary']),
                "education": row['education'],
                "experience": int(row['experience']),
                "willing_to_move": row['willing_to_move'] == 'True'
            })
    return cand_list
# Load employers data from CSV
def load_employers(f_name):
    emp_list = []
    
    with open(f_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            emp_list.append({
                "id": int(row['id']),
                "job_desc": row['job_desc'],
                "location": row['location'],
                "salary_min": int(row['salary_min']),
                "salary_max": int(row['salary_max']),
                "education": row['education'],
                "experience": int(row['experience']),
                "remote": row['remote'] == 'True'
            })
    return emp_list

# Function to calculate cosine similarity between job descriptions
def cal_similarity(job_1, job_2):
    # In a real-world scenario, use a proper NLP model or library to calculate semantic similarity
    # Here we're using a simple approach by comparing word frequencies
    w1 = job_1.split()
    w2 = job_2.split()
    all_words = list(set(w1 + w2))
    vector1 = [w1.count(word) for word in all_words]
    vector2 = [w2.count(word) for word in all_words]
    
    return cosine_similarity([vector1], [vector2])[0][0]

# Function to calculate rank score based on employer and candidate data
def cal_score(cand, emp):
    # Calculate similarity based on job description
    job_score = cal_similarity(cand['job_desc_exp'], emp['job_desc'])
    
    # Salary Score (normalized between 0 and 1)
    salary_score = 1 - abs(cand['salary'] - (emp['salary_min'] + emp['salary_max']) / 2) / max(emp['salary_max'], cand['salary'])
    
    # Experience Match (normalized between 0 and 1)
    exp_score = 1 - abs(cand['experience'] - emp['experience']) / max(emp['experience'], cand['experience'])
    
    # Education Match (simple boolean)
    edu_score = 1 if cand['education'] == emp['education'] else 0
    
    # Location and Willingness to Move
    loc_score = 1 if cand['location'] == emp['location'] or (cand['willing_to_move'] and emp['remote']) else 0
    
    # Final score (weighted average of all scores)
    total = 0.4 * job_score + 0.2 * salary_score + 0.2 * exp_score + 0.1 * edu_score + 0.1 * loc_score
    return total

# Function to rank candidates for each employer
def rank_candidates(cand, emp):
    rankings = {}
    
    for e in emp:
        emp_score_list = []
        
        for c in cand:
            score = cal_score(c, e)
            emp_score_list.append((c['id'], score))
        
        emp_score_list.sort(key=lambda x: x[1], reverse=True)
        rankings[e['id']] = emp_score_list
    
    return rankings

# Main function to execute the ranking
def main():
    cand = fetch_candidates('candidates.csv')
    emp = load_employers('employers.csv')
    
    rankings = rank_candidates(cand, emp)
    
    for emp_id, rank_cand in rankings.items():
        print(f"Employer {emp_id} Rankings:")
        for rank, (candidate_id, score) in enumerate(rank_cand, 1):
            print(f"Rank {rank}: Candidate {candidate_id}, Score: {score:.2f}")
        print("\n")

if __name__ == "__main__":
    main()
