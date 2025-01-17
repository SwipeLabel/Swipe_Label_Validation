import json
from typing import List, Dict, Any
from openai import OpenAI

class SwipeValidation:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(api_key=config['openai_key'])
        
    def _verify_image_content(self, image_url: str, expected_object: str) -> bool:
        try:
            # Create the prompt for GPT-4 Vision
            prompt = f"Is there a {expected_object.lower()} in this image? Please answer with just 'yes' or 'no'."
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=10
            )
            
            # Get the response and normalize it
            ai_response = response.choices[0].message.content.lower().strip()
            return 'yes' in ai_response
            
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            return False
    
    def validate(self, data: List[Dict[str, Any]]) -> float:
        if not data:
            return 0.0
        
        correct_validations = 0
        total_validations = len(data)
        
        for item in data:
            # Get the image URL and expected object from the data
            image_url = item['imgUrl']
            expected_object = item['imgText']
            user_response = bool(item['userResponse'])
            
            # Verify with GPT-4 Vision
            ai_sees_object = self._verify_image_content(image_url, expected_object)
            
            # Compare AI's response with user's response
            if ai_sees_object == user_response:
                correct_validations += 1
        
        # Calculate and return the accuracy score (0.0 to 1.0)
        return correct_validations / total_validations


if __name__ == "__main__":

    with open("Swipes.json", "r") as f:
        data = json.load(f)

    validator = SwipeValidation()
    results = validator.validate(data)
    print(f"Validation accuracy: {results:.2%}")
    