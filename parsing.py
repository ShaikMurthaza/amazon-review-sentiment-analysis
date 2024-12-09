import os
import json
import pandas as pd


class RobustJSONLParser:
    @staticmethod
    def parse_jsonl(file_path):
        """
        Robust JSONL parsing with error handling
        
        Args:
            file_path (str): Path to JSONL file
        
        Returns:
            list: Parsed review data
        """
        parsed_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    review = json.loads(line)
                    
                    if not isinstance(review, dict):
                        print(f"Warning: Line {line_num} is not a valid JSON object. Skipping.")
                        continue
                    
                    parsed_data.append(review)
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(f"Problematic line: {line}")
                    
                    try:
                        repaired_line = line.encode('unicode_escape').decode()
                        repaired_review = json.loads(repaired_line)
                        parsed_data.append(repaired_review)
                        print(f"Successfully repaired line {line_num}")
                    except Exception:
                        print(f"Could not repair line {line_num}")
                
                except Exception as e:
                    print(f"Unexpected error on line {line_num}: {e}")
        
        return parsed_data

    @staticmethod
    def convert_to_dataframe(parsed_data):
        """
        Convert parsed JSONL data to DataFrame
        
        Args:
            parsed_data (list): List of parsed review dictionaries
        
        Returns:
            pandas.DataFrame: Processed review data
        """
        data = []
        for review in parsed_data:
            full_text = f"{review.get('title', '')} {review.get('text', '')}".strip()
            
            data.append({
                'review_text': full_text,
                'rating': float(review.get('rating', 0.0)),
                'verified_purchase': bool(review.get('verified_purchase', False))
            })
        
        return pd.DataFrame(data)

def main():
    jsonl_path = 'automotive_1gb.jsonl'
    
    try:
        parsed_reviews = RobustJSONLParser.parse_jsonl(jsonl_path)
        
        df = RobustJSONLParser.convert_to_dataframe(parsed_reviews)
        
        print("Total reviews parsed:", len(df))
        print("Sample of parsed data:")
        print(df.head())
        
        df.to_csv('parsed_reviews.csv', index=False)
    
    except Exception as e:
        print(f"Error processing JSONL file: {e}")

if __name__ == "__main__":
    main()







