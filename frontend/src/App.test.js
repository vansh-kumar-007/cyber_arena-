import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

const memoryPayload = {
  success: true,
  data: {
    summary: {
      total_experiences: 3,
      successful_outcomes: 1,
      failed_outcomes: 1,
      active_lessons: 3,
      observed_success_rate: 0.5,
    },
    experiences: [],
  },
};

beforeEach(() => {
  global.fetch = jest.fn((url) => Promise.resolve({
    ok: true,
    json: () => Promise.resolve(String(url).includes("/memory") ? memoryPayload : { success: true, data: {} }),
  }));
});

afterEach(() => {
  jest.clearAllMocks();
});

test("renders the CyberArena game and agent-memory entry point", () => {
  render(<App />);
  expect(screen.getByRole("button", { name: /AGENT MEMORY & LEARNING/i })).toBeInTheDocument();
});

test("opens memory panel and displays persisted outcome metrics", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: /AGENT MEMORY & LEARNING/i }));
  expect(await screen.findByText("SUCCESSFUL OUTCOMES")).toBeInTheDocument();
  expect(screen.getByText("50.0%")).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining("/memory?limit=25"));
});
