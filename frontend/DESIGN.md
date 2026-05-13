---
name: Academic Minimalism
version: alpha
colors:
  primary: "#1A4F65"        # Deep teal/slate
  secondary: "#64748B"      # Slate gray
  tertiary: "#0EA5E9"       # Light sky blue accent
  neutral-bg: "#FFFFFF"     # Pure white
  surface: "#F8FAFC"        # Very light gray
  border: "#E2E8F0"         # Thin slate gray
  text-main: "#0F172A"      # Near black
  text-muted: "#64748B"
typography:
  h1:
    fontFamily: "Inter, sans-serif"
    fontSize: "2.5rem"
    fontWeight: 700
  h2:
    fontFamily: "Inter, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
  body:
    fontFamily: "Inter, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 400
  monospace:
    fontFamily: "'Fira Code', monospace"
    fontSize: "0.85rem"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  upload-zone:
    backgroundColor: "{colors.surface}"
    borderColor: "{colors.border}"
    textColor: "{colors.text-main}"
    rounded: "{rounded.lg}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.md}"
    padding: "12px 24px"
  button-primary-hover:
    backgroundColor: "#133a4b"
  table-header:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.primary}"
---

## Overview

Academic Minimalism focuses on clarity, precision, and high contrast. The UI evokes a modern scientific journal or a high-end data analytics platform. It uses stark white backgrounds, thin slate borders, and deep teal accents.

## Colors

The palette is rooted in pure whites and slates with a single deep teal accent for interactive elements.

- **Primary (#1A4F65):** Deep teal for primary actions and key branding.
- **Surface (#F8FAFC):** A very light gray used to separate interactive zones (like the upload box) from the pure white background.
- **Border (#E2E8F0):** Crisp, thin slate borders for structural delineation.
- **Text Main (#0F172A):** High-contrast near-black for readability.

## Typography

We use **Inter** for all UI elements to maintain a clinical, precise, and highly legible aesthetic. Monospace fonts are used exclusively for tabular numerical data and raw JSON.

## Shapes

Corners are slightly rounded (`6px` to `8px`) to soften the interface without losing the structured, grid-like feel of a scientific tool.
