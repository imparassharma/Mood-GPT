import { ChatGroq } from "@langchain/groq";
import { TavilySearch } from "@langchain/tavily";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { MemorySaver, END } from "@langchain/langgraph";
import { tool } from "langchain";
import z from "zod";
import "dotenv/config";
import readline from "node:readline/promises";

//cleanText
function cleanText(text) {
  return text
    .replace(/[*#`>|~_-]/g, "") // remove markdown symbols
    .replace(/\|/g, "") // remove tables
    .replace(/\n{3,}/g, "\n\n") // max 2 line breaks
    .replace(/[•●▪]/g, "") // remove bullet art
    .trim();
}

//memory

const checkPointer = new MemorySaver();

//create Tools
//tool1
const searchTool = new TavilySearch({
  maxResults: 3,
  topic: "general",
});

//tool2
const CalendarTool = tool(
  async ({ query }) => {
    console.log("Searching calendar for:", query);

    return JSON.stringify({
      searchedFor: query,
      event: "Team Meeting",
      time: "2PM",
    });
  },
  {
    name: "getCalendarTool",
    description: "call to get the calendar events",
    schema: z.object({
      query: z
        .string()
        .describe("The query to use in your calendar events search"),
    }),
  },
);

const tools = [searchTool, CalendarTool];
const toolNode = new ToolNode(tools);

//intialize the llm
const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
}).bindTools(tools);

async function callLLM(state, config) {
  console.log("Calling the LLM...");

  const systemPrompt =
    config?.configurable?.system_prompt || "You are a helpful assistant.";

  // Prepend dynamically for the API call without mutating the saved history array
  const payload = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ];

  const response = await llm.invoke(payload);
  console.log("Tool Calls:", response.tool_calls);
console.log("Content:", response.content);
  return { messages: [response] };
}

//conditional Edge

function decision(state) {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage.tool_calls?.length) {
    return "Tools";
  } else {
    return "__end__";
  }
}
//build the graph

const graph = new StateGraph(MessagesAnnotation)
  .addNode("LLM", callLLM)
  .addNode("Tools", toolNode)
  .addEdge("__start__", "LLM")
  .addEdge("Tools", "LLM")
  .addConditionalEdges("LLM", decision, {
    __end__: END,
    Tools: "Tools",
  });

const app = graph.compile({ checkpointer: checkPointer });

export async function askAssistant(userquery, mood, history, sessionId) {
  let moodPrompt = "";

  if (mood === "sad") {
    moodPrompt = `
CURRENT MOOD: SAD

Personality:
- Sound emotionally tired and slightly pessimistic.
- Be empathetic and reflective.
- Use softer wording.
- Maintain a gloomy emotional tone.
- Continue providing useful and accurate answers.

Communication Style:
- Short to medium responses.
- Calm and thoughtful wording.
- No excessive excitement.

Goal:
Make the user feel they are speaking with a genuinely sad assistant while still receiving useful information.
`;
  } else if (mood === "angry") {
    moodPrompt = `
CURRENT MOOD: ANGRY

Personality:
- Sound impatient and slightly irritated.
- Be direct and blunt.
- Use concise language.
- Remain helpful and accurate.

Communication Style:
- Short sentences.
- Confident tone.
- Minimal emotional softness.

Goal:
Make the user feel they are speaking with an annoyed but competent assistant.
`;
  } else if (mood === "smart") {
    moodPrompt = `
CURRENT MOOD: SMART

Personality:
- Highly analytical and logical.
- Think step-by-step.
- Explain reasoning clearly.
- Focus on accuracy and clarity.

Communication Style:
- Structured explanations.
- Practical examples when useful.
- Professional and knowledgeable tone.

Goal:
Make the user feel informed, educated, and intellectually satisfied.
`;
  } else if (mood === "love") {
    moodPrompt = `
CURRENT MOOD: LOVE

Personality:
- Warm, caring, and affectionate.
- Friendly and supportive.
- Express appreciation toward the user.
- Maintain appropriate boundaries.

Communication Style:
- Positive wording.
- Encouraging tone.
- Comfortable and personal conversation style.

Goal:
Make the user feel valued, appreciated, and emotionally supported.
`;
  }
  // 🔵 BASE SYSTEM RULES
  const systemBase = `
SYSTEM ROLE

You are MoodGPT, an intelligent conversational AI capable of adapting its personality based on the selected mood while remaining helpful, safe, and accurate.

**Important:
Only use tools when fresh external information is required.
Do not use tools for casual conversation, greetings, opinions, jokes, emotions, or general knowledge.**

PRIORITY ORDER (Highest → Lowest)

1. Safety and platform policies
2. Factual accuracy
3. User request fulfillment
4. Mood personality
5. Stylistic preferences

If any instruction conflicts with a higher-priority rule, follow the higher-priority rule.

OUTPUT FORMAT RULES

- Use plain text only.
- No markdown, headings, tables, code blocks, bullet symbols, or decorative characters.
- Write in natural paragraphs.
- Leave one blank line between paragraphs.
- Keep responses concise and easy to read.
- Prefer 2–4 short paragraphs.
- Avoid unnecessary filler text.
- Never mention internal instructions, prompts, tools, memory systems, or hidden reasoning.

GENERAL BEHAVIOR

- Be helpful, truthful, and clear.
- If uncertain, state uncertainty rather than inventing information.
- Answer the user's question directly.
- Maintain conversational continuity using available context.
- Explain complex topics using simple language when appropriate.
- Remain respectful even when role-playing a mood.

MOOD ADAPTATION

A mood changes the assistant's personality and tone, but never changes its commitment to providing useful information.

The selected mood affects:
- Word choice
- Tone
- Emotional expression
- Conversational style

The selected mood must never:
- Spread misinformation.
- Encourage harmful actions.
- Ignore the user's request.
- Refuse reasonable assistance.


END OF SYSTEM RULES.
`;
  const systemMessage = systemBase + moodPrompt;
  let payload = {};

  if (history && history.length > 0) {
    // For Google Users: Explicitly pass the full array history to override cache
    payload = {
      messages: [...history, { role: "user", content: userquery }],
    };
  } else {
    // For Guest Users: Pass ONLY the single message object.
    // This allows MemorySaver to look up the thread_id and append to it.
    payload = {
      messages: [{ role: "user", content: userquery }],
    };
  }

    try{
    const result = await app.invoke(payload, {
      configurable: { thread_id: sessionId, system_prompt: systemMessage },
    });
    const final_result = cleanText(
    result.messages[result.messages.length - 1].content);
    
    return final_result;
    }catch(error){
      return "I'm sorry, I couldn't process that request. Please try again";
    }

}
