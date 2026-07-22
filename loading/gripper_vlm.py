import os
import json
from PIL import Image
from google import genai
from google.genai import types


client = genai.Client(api_key="AQ.Ab8RN6JhVNsixZ6ipoYug3pBPnFVVpzkvFihaUOj2j6sVYBHBQ")


def get_vlm_predictions(image_path):
    """Sends the image to the VLM to extract mass and friction estimates."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Please place your test image at: {image_path}")
        
    img = Image.open(image_path)
    
    prompt = """
    Analyze the object in this image intended for a robotic two-fingered gripper. 
    Estimate two values based on its visual traits:
    1. The total mass (mass_kg) in kilograms.
    2. The static coefficient of friction (friction_coefficient) between the object 
       surface and a standard flat rubber gripper pad (typically between 0.15 and 0.8).
       
    You must return a raw JSON object matching this schema:
    {"mass_kg": float, "friction_coefficient": float}
    Do not output markdown formatting, wrappers, or text explanations.
    """
    
    print("Sending image to VLM for physical parameter inference...")
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=[img, prompt],
        # Enforce that the model can only reply with valid, structured JSON
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text

def calculate_physics_force(vlm_json_string, safety_factor=1.5):
    """Applies Coulomb's law using the parameters provided by the VLM."""
    parsed_data = json.loads(vlm_json_string)
    m = parsed_data["mass_kg"]
    mu = parsed_data["friction_coefficient"]
    g = 9.81  # Gravity acceleration in m/s^2
    
    # Coulomb's friction law for a parallel 2-fingered gripper:
    # Fn >= (m * g) / (2 * mu)
    required_normal_force = ((m * g) / (2 * mu)) * safety_factor
    
    print("\n================ PHYSICS CONTROLLER OUTPUT ================")
    print(f"Visual Object Identification Feedback:")
    print(f"  • Estimated Mass:            {m} kg ({m*1000:.1f} grams)")
    print(f"  • Estimated Friction (μ):     {mu}")
    print(f"  • Applied Safety Multiplier:  {safety_factor}x")
    print(f"-----------------------------------------------------------")
    print(f"Target Clamping Force:       {required_normal_force:.2f} Newtons")
    print("===========================================================\n")

if __name__ == "__main__":
    # Ensure your image filename matches this string
    IMAGE_FILE = "images/Box.jpg" 
    
    try:
        raw_vlm_output = get_vlm_predictions(IMAGE_FILE)
        calculate_physics_force(raw_vlm_output)
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")
