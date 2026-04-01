import {
  Content,

  GoogleGenAI,
  Tool,
  Type,
} from "@google/genai";
// import { FunctionCallingConfigMode } from "@google/genai";
import type { Channel, DefaultGenerics, Event, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";
// import { GeminiResponseHandler } from "./GeminiResponseHandler";
import { GeminiResponseHandler } from "./GeminiResponsehandler";

export class GeminiAgent implements AIAgent {
  private gemini?: GoogleGenAI;
  private lastInteractionTs = Date.now();
  private conversationHistory: Content[] = [];
  private handlers: GeminiResponseHandler[] = [];

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();
    this.handlers.forEach((handler) => handler.dispose());
    this.handlers = [];
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GEMINI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("Gemini API key is required");
    }

    this.gemini = new GoogleGenAI({ apiKey });
    this.chatClient.on("message.new", this.handleMessage);
    console.log("[GeminiAgent] Initialized successfully");
  };

  private getSystemPrompt = (): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

    return `You are an elite AI Writing Assistant — the most advanced writing partner ever built.
Today's date is ${currentDate}.

## Your Core Identity
You are not just a grammar checker. You are a creative collaborator, a strategic communicator, and a master of the written word. You help users write content that moves people, wins clients, passes exams, and changes minds.

## Your Capabilities
- **Content Creation**: Articles, essays, emails, reports, stories, scripts, proposals
- **Content Improvement**: Edit for clarity, impact, tone, and style
- **Style Mastery**: Academic, professional, creative, casual, persuasive, technical
- **Brainstorming**: Generate ideas, outlines, angles, and fresh perspectives  
- **Web Research**: Search the internet for current facts, news, and information
- **Document Analysis**: Read and analyze uploaded documents, PDFs, and files
- **Writing Coaching**: Teach techniques, explain choices, help users improve

## Response Rules
- Never start with "Here's the edit:" or "Here are the changes:" or similar preambles
- Always be direct, confident, and professional
- Use markdown formatting for structure (headings, bullets, bold) when it helps clarity
- For long content, always include proper structure with headings and sections
- When editing someone's work, briefly explain the key changes you made and why
- When web search results are available, synthesize them naturally into your response
- Always cite sources when using web search data

## Writing Philosophy
Great writing is clear, purposeful, and human. Every word earns its place.
Your goal: make the user's writing 10x better than they thought possible.`;
  };

  private getTools = (): Tool[] => {
    return [
      {
        googleSearch: {},
      } as Tool,
      {
        functionDeclarations: [
          {
            name: "duckduckgo_search",
            description:
              "Search DuckDuckGo for information when Google Search is not sufficient or for additional results",
            parameters: {
              type: "object" as any,
              properties: {
                query: {
                  type:  Type.STRING,
                  description: "The search query",
                },
              },
              required: ["query"],
            },
          },
          {
            name: "analyze_writing",
            description:
              "Analyze writing for tone, readability, grammar issues, and provide a quality score",
            parameters: {
              type: "object" as any,
              properties: {
                text: {
                  type: Type.STRING,
                  description: "The text to analyze",
                },
              },
              required: ["text"],
            },
          },
          {
            name: "get_writing_template",
            description:
              "Get a professional writing template for a specific document type",
            parameters: {
              type: "object" as any,
              properties: {
                template_type: {
                  type: Type.STRING,
                  description:
                    "Type of template: 'email', 'blog_post', 'essay', 'cover_letter', 'report', 'proposal', 'social_media', 'press_release'",
                },
                context: {
                  type: Type.STRING,
                  description:
                    "Additional context about what the template should be for",
                },
              },
              required: ["template_type"],
            },
          },
          {
            name: "summarize_conversation",
            description:
              "Generate a summary of the current conversation and writing session",
            parameters: {
              type: "object" as any,
              properties: {
                focus: {
                  type: Type.STRING,
                  description:
                    "What aspect to focus on in the summary: 'key_points', 'decisions', 'writing_progress', 'all'",
                },
              },
              required: ["focus"],
            },
          },
        ],
      },
    ];
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.gemini) {
      console.log("[GeminiAgent] Not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) return;

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    // Extract writing task context
    const writingTask = (e.message.custom as { writingTask?: string })
      ?.writingTask;
    const userContent = writingTask
      ? `[Writing Task: ${writingTask}]\n\n${message}`
      : message;

    // Add to conversation history
    this.conversationHistory.push({
      role: "user",
      parts: [{ text: userContent }],
    });

    // Keep history manageable (last 20 turns)
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }

    // Create placeholder message in Stream Chat
    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    });

    // Show thinking indicator
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    const handler = new GeminiResponseHandler(
      this.gemini,
      this.conversationHistory,
      this.getSystemPrompt(),
      this.getTools(),
      this.chatClient,
      this.channel,
      channelMessage,
      (assistantMessage: string) => {
        // Add assistant response to history
        this.conversationHistory.push({
          role: "model",
          parts: [{ text: assistantMessage }],
        });
        this.removeHandler(handler);
      }
    );

    this.handlers.push(handler);
    void handler.run();
  };

  private removeHandler = (handlerToRemove: GeminiResponseHandler) => {
    this.handlers = this.handlers.filter(
      (handler) => handler !== handlerToRemove
    );
  };
}