/**
 * API client for the Agentic Banking backend.
 * All requests include the thread_id for session continuity.
 */

import { getThreadId } from "./threadId";

const API_BASE = "/api/v1";

interface ChatResponse {
  reply: string;
  thread_id: string;
}

interface ManagerChatResponse {
  reply: string;
  is_waiting?: boolean;
  thread_id?: string;
}

/**
 * Send a message to the Bank Manager agent (supervisor/router).
 */
export async function chatWithManager(message: string): Promise<ManagerChatResponse> {
  const threadId = getThreadId();
  const res = await fetch(`${API_BASE}/manager/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!res.ok) throw new Error(`Manager chat failed: ${res.status}`);
  return res.json();
}

/**
 * Send a message to the Privacy Policy agent.
 */
export async function chatWithPrivacy(message: string): Promise<ChatResponse> {
  const threadId = getThreadId();
  const res = await fetch(`${API_BASE}/privacy/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!res.ok) throw new Error(`Privacy chat failed: ${res.status}`);
  return res.json();
}

/**
 * Submit an admin review decision (approve/reject) for a pending transaction.
 */
export async function reviewTransaction(
  threadId: string,
  decision: "approve" | "reject"
): Promise<{ message: string; final_status: string }> {
  const res = await fetch(`${API_BASE}/admin/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, decision }),
  });
  if (!res.ok) throw new Error(`Review failed: ${res.status}`);
  return res.json();
}

export interface Transaction {
  icon: string;
  name: string;
  type: string;
  amount: string;
  status: string;
  statusIcon: string;
  statusColor: string;
}

export interface DashboardData {
  full_name: string;
  balance: number;
  kyc_status: string;
  recent_transactions: Transaction[];
}

/**
 * Fetch the dashboard data for a specific user.
 * We are hardcoding user_id = 1 for the "Alice Protocol" phase.
 */
export async function getDashboard(userId: number = 1): Promise<DashboardData> {
  const res = await fetch(`${API_BASE}/user/dashboard/${userId}`);
  if (!res.ok) throw new Error(`Failed to fetch dashboard: ${res.status}`);
  return res.json();
}
