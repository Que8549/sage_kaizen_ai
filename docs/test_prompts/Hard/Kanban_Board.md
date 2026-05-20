# Kanban Board with Drag & Drop

Create a single self-contained `index.html` file — a fully functional Kanban board for personal task management.

## Core Features

### Columns
- Start with **four default columns**: `Backlog`, `To Do`, `In Progress`, `Done`
- Allow the user to **add a new custom column** by clicking an "＋ Add Column" button at the end of the board
- Allow the user to **rename any column** by double-clicking its header
- Allow the user to **delete a column** (only if it is empty) via a trash icon that appears on hover

### Cards
Each card must support:
- **Title** (required) and optional **Description** (shown as a collapsible detail below the title)
- **Priority badge**: `Low` / `Medium` / `High` — each with a distinct color (e.g. green / amber / red)
- **Due date** field — if the due date is in the past, display it in red with a ⚠️ icon
- **Delete button** — appears on card hover, removes the card with a fade-out animation

Add a card by clicking an "＋ Add card" button at the bottom of any column. Show an inline mini-form (title input + priority selector + due date picker + optional description textarea + Save/Cancel buttons) — do NOT use browser `prompt()`.

### Drag & Drop
- Cards must be **draggable between columns** using the native HTML5 Drag and Drop API (no external libraries)
- While dragging, highlight the target column with a visible drop zone indicator
- Moving a card to the `Done` column should automatically update the card's visual style (e.g. muted colors, strikethrough title)

## Filtering & Search
- A **search bar** at the top of the board filters cards in real time by title (case-insensitive)
- A **priority filter** dropdown (`All`, `Low`, `Medium`, `High`) hides cards that don't match; hidden cards should leave a placeholder gap so the column height doesn't jump

## Persistence
- The **entire board state** (columns, cards, their order) must be saved to `localStorage` automatically after every change
- On page load, restore the saved state; if no saved state exists, populate the board with a small set of example cards spread across all four default columns

## Statistics Bar
A fixed bar at the very bottom of the viewport showing live-updating numbers:
- **Total cards** across the board
- **Overdue cards** (due date < today)
- **Done** (count of cards in the `Done` column)
- **Completion rate** as a percentage: `Done / Total × 100%`

## Design Requirements
- **Dark theme** — deep charcoal/slate background (`#0f1117` or similar)
- **Glassmorphism column cards** — semi-transparent frosted-glass panels with a subtle border and shadow
- **Modern font** — load `Inter` from Google Fonts
- Color-coded priority badges with soft pill shape
- Smooth transitions: card hover lift, drag ghost opacity, column drop-zone pulse
- Fully responsive — columns should scroll horizontally on small screens without breaking layout

## Technical Constraints
- Single `index.html` file — HTML, CSS, and JavaScript all inline
- Vanilla JavaScript only — no frameworks, no external JS libraries
- Google Fonts CDN is allowed
- The file must work correctly by simply opening it in a browser (no server needed)
- Code must be well-structured with clear comments separating major sections (State, Rendering, Drag & Drop, Persistence, etc.)
