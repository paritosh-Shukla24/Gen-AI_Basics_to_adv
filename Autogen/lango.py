import base64
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import OllamaLLM

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# def plt_img_base64(img_base64):
#     """
#     Display base64 encoded string as image
#
#     :param img_base64:  Base64 string
#     """
#     # Create an HTML img tag with the base64 string as the source
#     image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
#     # Display the image by rendering the HTML
#     display(HTML(image_html))


file_path = "NOps (1).jpg"
pil_image = Image.open(file_path)
image_b64 = convert_to_base64(pil_image)
# plt_img_base64(image_b64)
prompt = ChatPromptTemplate.from_template("Youu will Show Some Image Relayted to Hackathon Describe it and Create Story Out of It how Many Peoople Was Present and they Loosed Smart India Hackathon And Describe it more")





llm = OllamaLLM(model="llama3.2-vision",
                format="json")
chain = prompt | llm
llm_with_image_context = llm.bind(images=[image_b64])
res=llm_with_image_context.invoke("Youu will Show Some Image Relayted to Hackathon Describe it and Create Story Out of It how Many Peoople Was Present and they Loosed Smart India Hackathon And Describe it more")
print(res)