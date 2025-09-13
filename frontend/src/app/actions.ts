"use server";

// AI functionality temporarily disabled
export async function getSuggestedReply(
  patientMessage: string,
  context?: string
) {
  return {
    suggestedReply: "AI feature is currently under development.",
    reasoning: "This feature will be available soon.",
  };
}
