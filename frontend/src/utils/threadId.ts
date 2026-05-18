/**
 * Thread ID utility for maintaining unique user sessions.
 * 
 * Each browser session gets a unique thread_id stored in localStorage.
 * This ensures the backend (LangGraph) can distinguish between different
 * users/sessions and keep their conversation state separate.
 */

const THREAD_ID_KEY = "chat_thread_id";

/**
 * Get or create a unique thread ID for this browser session.
 * Uses crypto.randomUUID() — available in all modern browsers.
 */
export function getThreadId(): string {
  let threadId = localStorage.getItem(THREAD_ID_KEY);
  if (!threadId) {
    threadId = crypto.randomUUID();
    localStorage.setItem(THREAD_ID_KEY, threadId);
  }
  return threadId;
}

/**
 * Force-reset the thread ID (e.g., for a "New Conversation" button).
 * Returns the new thread ID.
 */
export function resetThreadId(): string {
  const threadId = crypto.randomUUID();
  localStorage.setItem(THREAD_ID_KEY, threadId);
  return threadId;
}
