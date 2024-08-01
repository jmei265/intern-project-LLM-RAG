# Offensive Cyber Assistant

## Project Description

The Offensive Cyber Assistant project develops a Retrieval-Augmented Generation (RAG) Large Language Model (LLM) that is designed specifically for offensive cybersecurity professionals such as red teamers and penetration testers. This model leverages advanced language processing to provide insights into vulnerabilities and weaknesses that can be exploited in various systems.

By using the RAG LLM, security professionals can:

- **Identify Vulnerabilities:** Gain insights into potential vulnerabilities within a given system or application.
- **Understand Exploits:** Receive detailed information on known exploits and how they can be leveraged in penetration testing scenarios.
- **Enhance Red Team Exercises:** Utilize the model to simulate attacks and assess security measures effectively.
- **Stay Informed:** Keep up with the latest threat vectors and security practices to enhance overall cybersecurity strategies.

The RAG LLM combines up-to-date knowledge with powerful analysis capabilities to support proactive security measures and improve the effectiveness of security assessments.

## Features
- List key features of the project
- Highlight what makes it unique
- Mention any notable functionalities
  
## Installation
Instructions on how to set up this project:
1. Setup GPU instance:
   - Follow the documentation to set up anaconda and create an EC2 instance and install a Ubuntu virtual machine on it.
3. Install Langchain framework
   - In VSCode, go to Terminal, then use the command pip install<library> to installl the libraries from the requirements.txt file
5. Install Llama3
   - In your script, run the following code:
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
        os.system("export OLLAMA_HOST=localhost:8888")
        os.system("sudo service ollama stop")
        cmd = "ollama serve"
        with open(os.devnull, 'wb') as devnull:
            process = subprocess.Popen(cmd, shell=True, stdout=devnull, stderr=devnull)
7. Install WhiteRabbitNeo
   - Run the following code:
        os.system("ollama pull jimscard/whiterabbit-neo")
        llm = Ollama(model="jimscard/whiterabbit-neo")
        return llm
9. Install Filezilla on VM
    - Follow the SOP in the documentation for correct installation.

## Technologies Used
![image](https://github.com/user-attachments/assets/38692311-80b4-4ec6-afa3-cd8a505be714)
These can be installed from our requirements.txt file.


## Contributing
We welcome contributions to this project! To contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the Llama3 License. For more details, refer to the LICENSE file in the repository.

## Authors and Acknowledgments
This project was developed by the intern team at Everwatch Corporation:

- Myra Cropper
- Jonathan Mei
- Sachin Ashok
- Jonathan Rogers
- Tanvi Simhadri
- Kyle Simon
- Connor Mullikin
- Matthew Lessler
- Izaiah Davis
- Quinn Dunnigan

Mentored by David Culver.

## Contact Information
For questions or issues, please open an issue in the repository or contact the authors.
