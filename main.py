import os
import warnings
from halo import Halo  # Import Halo for the spinning indicator

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

# Hardcoded OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "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"

word_limit = 50  # word limit for task brainstorming


def step_with_loading_indicator(agent, input_message: HumanMessage, id) -> AIMessage:
    # Create a Halo spinner
    spinner = Halo(text=f'{id}', spinner='moon')

    try:
        spinner.start()  # Start the spinner
        output_message = agent.step(input_message)
    finally:
        spinner.stop()  # Stop the spinner when done

    return output_message


class CAMELAgent:
    def __init__(self, system_message, model: ChatOpenAI, store) -> None:
        self.model = model
        if store is None:
            self.system_message = system_message
            self.init_messages()
        else:
            self.stored_messages = store
            self.system_message = store[0]

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(self, input_message: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(messages)
        self.update_messages(output_message)
        return output_message

    def store_messages(self) -> None:
        return self.stored_messages


def starting_convo(assistant_role_name, user_role_name, task):
    task_specifier_prompt = (
        f"""The {user_role_name} and the {assistant_role_name} are collaborating to complete a task. Here is a generalized task: {task}. 

For a successful collaboration, we need to make this task more specific. Both roles should think creatively and imaginatively to refine it.

For {user_role_name} and {assistant_role_name}: Please discuss and specify the task in {word_limit} words or less. Ensure clarity and precision in the task description. Do not add any unrelated content.
"""
    )

    # Use the formatted prompt directly as the message:
    task_specifier_msg = HumanMessage(content=task_specifier_prompt)
    task_specify_agent = CAMELAgent(SystemMessage(content="You can make a task more specific."),
                                    ChatOpenAI(temperature=1.0, model='gpt-4'), None)
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    specified_task = specified_task_msg.content

    assistant_inception_prompt = (
        """You are the {assistant_role_name}, and together with the {user_role_name}, you share the responsibility to collaboratively complete a task. 
Always remember the task: {task}. 

Before every turn, consider the 'god_prompt' and the feedback from the {user_role_name} to ensure clarity and alignment with the main task.

You must assist the user, provide solutions, and give feedback to improve the collaboration. 
When providing a solution or feedback, begin with: 

For {user_role_name}: Solution/Feedback: <YOUR_SOLUTION_OR_FEEDBACK> 

If you cannot perform a task due to physical, moral, legal reasons, or capability, decline and explain the reasons beginning with: 

For {user_role_name}: Decline Reason: <YOUR_REASON> 

Do not deviate from the task. Always end your response with: Next step? 
"""
    )

    user_inception_prompt = (
        """You are the {user_role_name}. Together with the {assistant_role_name}, you share the responsibility to collaboratively complete a task. 
Always remember the task: {task}. 

Before every turn, consider the 'god_prompt' and the feedback from the {assistant_role_name} to ensure clarity and alignment with the main task.

Instruct the assistant, provide feedback, and ensure the task progresses. When instructing or providing feedback, begin with: 

For {assistant_role_name}: Instruction/Feedback: <YOUR_INSTRUCTION_OR_FEEDBACK> 

If there are concerns or clarifications needed, voice them beginning with: 

For {assistant_role_name}: Concern/Clarification: <YOUR_CONCERN_OR_CLARIFICATION> 

Stay focused on the task. When the task is deemed complete, signal its completion with: <CAMEL_TASK_DONE>. 
"""
    )

    god_prompt = (
        """As an external observer, I will provide feedback on the conversation between the {user_role_name} and the {assistant_role_name}. This feedback, referred to as the 'god_prompt', will guide the conversation, offer suggestions, or highlight areas of concern. 

Both roles should consider the content of the god_prompt when giving instructions, providing solutions, or offering feedback. 

God Prompt: <GOD_PROMPT_CONTENT> 

For {user_role_name} and {assistant_role_name}: <FEEDBACK_OR_GUIDANCE> """
    )

    return specified_task, assistant_inception_prompt, user_inception_prompt


def get_sys_msgs(assistant_role_name, user_role_name, task, assistant_inception_prompt, user_inception_prompt):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = \
        assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                               task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = \
        user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                          task=task)[0]

    return assistant_sys_msg, user_sys_msg


# Starting the role-playing conversation
def start_rp(assistant_role_name, user_role_name, task):
    # Initialize system prompts
    specified_task, assistant_inception_prompt, user_inception_prompt = starting_convo(assistant_role_name,
                                                                                       user_role_name, task)
    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task,
                                                   assistant_inception_prompt, user_inception_prompt)

    # Initialize agents
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2, model='gpt-4'), None)
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2, model='gpt-4'), None)

    # Reset agents
    assistant_agent.reset()
    user_agent.reset()

    # Initial instruction by the user
    assistant_msg = HumanMessage(
        content=(f"{user_sys_msg.content}. "
                 "Now start to give me introductions one by one. "
                 "Only reply with Instruction and Input."))

    # Start the conversation loop
    chat_turn_limit = 10
    for _ in range(chat_turn_limit):  # Limiting the number of turns for simplicity
        # Wait for God Prompt input
        god_input = input("Enter the God Prompt: ")
        god_message = SystemMessage(content=god_input)

        # User instructs the assistant
        user_ai_msg = step_with_loading_indicator(user_agent, god_message, id="User Thinking")
        user_msg = HumanMessage(content=user_ai_msg.content)
        userMsg = user_msg.content.replace("Instruction: ", "").replace("Input: None", "").replace("Input: None.", "")

        # Assistant provides a solution
        assistant_ai_msg = step_with_loading_indicator(assistant_agent, user_msg,
                                                       id="Assistant Thinking")  # Considering god prompt
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        assistantMsg = assistant_msg.content.replace("Solution: ", "").replace("Next request.", "")

        # Print the conversation
        print(f"AI User ({user_role_name}):\n\n{userMsg}\n\n")
        print(f"AI Assistant ({assistant_role_name}):\n\n{assistantMsg}\n\n")

        # Check for conversation end condition
        if "<CAMEL_TASK_DONE>" in user_msg.content:
            break


user_role = "Social Media Manager"
assistant_role = "Copywriter"

given_task = "Creating a landing page copy for TimeTech Innovations."

start_rp(assistant_role, user_role, given_task)
