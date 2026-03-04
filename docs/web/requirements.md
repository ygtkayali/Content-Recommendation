# Web Requirements (v0)

## Functional Requirements
- Users can search/select a movie or TV title and request similar recommendations.
- Users can submit natural-language prompts for intent-based recommendations.
- Users can view recommendation cards with title, type, score/reason, and metadata.
- Users can handle empty, loading, and error states clearly.

## Non-Functional Requirements
- Responsive UI (desktop first, mobile supported).
- Basic accessibility (semantic HTML, keyboard navigation, readable contrast via existing theme).
- Input validation before API call.
- Clear error messaging for invalid input, unavailable results, and rate-limited responses.

## Team Deliverables
- Frontend architecture notes (components, route strategy, state management).
- UI wireframe for both recommendation flows.
- API client module with typed request/response contracts.
