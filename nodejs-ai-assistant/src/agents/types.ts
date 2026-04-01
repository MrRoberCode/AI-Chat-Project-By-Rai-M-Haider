import type { Channel, StreamChat, User } from "stream-chat";

export interface AIAgent {
  user?: User;
  channel: Channel;
  chatClient: StreamChat;
  getLastInteraction: () => number;
  init: () => Promise<void>;
  dispose: () => Promise<void>;
}

export enum AgentPlatform {
  GEMINI = "gemini",
  WRITING_ASSISTANT = "writing_assistant",
}

export interface WritingMessage {
  custom?: {
    messageType?: "user_input" | "ai_response" | "system_message";
    writingTask?: string;
    suggestions?: string[];
  };
}

export interface DocumentAnalysisRequest {
  base64Data: string;
  mimeType: string;
  fileName: string;
  prompt?: string;
  channelId: string;
  channelType?: string;
}

export interface ChatSummaryRequest {
  channelId: string;
  channelType?: string;
  messageLimit?: number;
}