import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import random
from instagrapi import Client
import os
import random
from langchain_huggingface import HuggingFaceEndpoint

attires = [
    # Traditional Attires
    "Saree",
    "Lehenga",
    "Salwar Kameez",
    "Anarkali Suit",
    "Kurti with Churidar",
    "Sharara Suit",
    "Patiala Suit",
    
    # Contemporary/Fusion Attires
    "Pant Saree",
    "Dhoti Saree",
    "Palazzo Suit",
    "Peplum Top with Dhoti Pants",
    "Indo-Western Gown",
    "Jumpsuit with Indian Prints",
    "Kurti with Jeans",
    "Asymmetrical Tunic with Leggings",
    "Culottes with Ethnic Blouse",
    "indo-western Lehenga",
    
    # Western Attires
    "Cocktail Dress",
    "Evening Gown",
    "Pencil Skirt with Blouse",
    "Maxi Dress",
    "Bodycon Dress",
    "Jumpsuit",
    "Blazer with Trousers",
    "Denim Jacket with Skirt",
    "Off-Shoulder Top with Jeans",
    "Wrap Dress",
    "Slip Dress",
    "Backless Dress",
    "Plunge Neckline Dress",
    "Halter Neck Dress"
]

locations = [
    # Urban/City Locations
    "New York City, USA",
    "Paris, France",
    "London, UK",
    "Tokyo, Japan",
    "Dubai, UAE",
    "Barcelona, Spain",
    "Hong Kong, China",
    "Los Angeles, USA",
    "Mumbai, India",
    "Singapore",
    
    # Nature/Scenic Locations
    "Bali, Indonesia",
    "Santorini, Greece",
    "Banff National Park, Canada",
    "Amalfi Coast, Italy",
    "Swiss Alps, Switzerland",
    "Great Barrier Reef, Australia",
    "Machu Picchu, Peru",
    "Yosemite National Park, USA",
    "Maldives",
    "Bora Bora, French Polynesia",
    
    # Historical/Cultural Locations
    "Rome, Italy",
    "Kyoto, Japan",
    "Istanbul, Turkey",
    "Petra, Jordan",
    "Cairo, Egypt",
    "Athens, Greece",
    "Angkor Wat, Cambodia",
    "Jaipur, India",
    "Jerusalem, Israel",
    "Marrakech, Morocco",
    
    # Tropical/Beach Locations
    "Miami, USA",
    "Phuket, Thailand",
    "Rio de Janeiro, Brazil",
    "Gold Coast, Australia",
    "Seychelles",
    "Tahiti, French Polynesia",
    "Mykonos, Greece",
    "Zanzibar, Tanzania",
    "Honolulu, USA",
    "Fiji",
    
    # Luxury/Modern Locations
    "Monte Carlo, Monaco",
    "Beverly Hills, USA",
    "Dubai Marina, UAE",
    "St. Barts, Caribbean",
    "Ibiza, Spain",
    "Maldives Overwater Villas",
    "Lake Como, Italy",
    "Aspen, USA",
    "Bora Bora, French Polynesia",
    "Parisian Rooftops, France",
    
    # Exotic/Adventure Locations
    "Amazon Rainforest, Brazil",
    "Sahara Desert, Morocco",
    "Patagonia, Argentina",
    "Icelandic Glaciers, Iceland",
    "Serengeti, Tanzania",
    "Galápagos Islands, Ecuador",
    "Great Wall of China, China",
    "Outback, Australia",
    "Rocky Mountains, Canada",
    "Cappadocia, Turkey"
]



# Set environment variable for Hugging Face API token

username = os.environ['username']
password = os.environ['password']



#lines = output.split('\n')
#print(lines)
#current = f"a Bollywood actress wearing {attire} in {location}"  + lines[3]


# Initialize the base model and specific LoRA
base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

lora_repo = "XLabs-AI/flux-RealismLora"
trigger_word = ""  # Leave trigger_word blank if not used.
pipe.load_lora_weights(lora_repo)

pipe.to("cuda")

MAX_SEED = 2**32-1

cfg_scale = 3.2
steps = 28
width = 1024
height = 1024
seed = random.randint(0, MAX_SEED)
lora_scale = 0.85


@spaces.GPU(duration=80)
def run_lora():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # Initialize the Hugging Face endpoint
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.2, token=os.environ['HF_TOKEN'])

    location = random.choice(locations)
    attire = random.choice(attires)
    output = llm.invoke(f"create text-to-image scene of bollywood actress at {location} in {attire}, keep the description short")
    caption = llm.invoke(f"{output}. create insta caption and 8 tags")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Update progress bar (0% saat mulai)


    # Generate image with progress updates
    # for i in range(1, steps + 1):
    #     # Simulate the processing step (in a real scenario, you would integrate this with your image generation process)
    #     if i % (steps // 10) == 0:  # Update every 10% of the steps
    #         progress(i / steps * 100, f"Processing step {i} of {steps}...")

    # Generate image using the pipeline
    image = pipe(
        prompt=output,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale},
    ).images[0]
    

    cl = Client()
    cl.login(username, password)

    image.save("generated_image.png")
    cl.photo_upload("generated_image.png", caption) 
    cl.logout()
with gr.Blocks() as app:
    gr.Markdown("# Flux RealismLora Image Generator")
    generate_button = gr.Button("Generate")
    generate_button.click(run_lora)
app.queue()
app.launch()

#Final update (100%)
