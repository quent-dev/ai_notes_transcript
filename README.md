# Basic transcription app using the AI21 Jamba 1.5 API.

## Setup

1. Create a `.env` file with the following:

```
ACTIVATE_KEY=AI21  # Options: OpenAI, AI21
AI21_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
```

## API Documentation
https://docs.ai21.com/reference/jamba-15-api-ref
https://supabase.com/docs/reference/python/insert


## Supported Languages

The app currently supports the following languages for transcription:

- English
- Spanish
- French

## Supabase

The app uses Supabase for database storage. The database schema is as follows:

- `notes`: This table stores the notes with the following columns:
  - `id`: The unique identifier for the note.
  - `content`: The content of the note.
  - `cost`: The cost of the transcription.
  - `created_at`: The timestamp when the note was created.

## To do

- [X] Add a column to the `notes` table to store the transcription cost.
- [X] Add option to view past notes.
- [ ] Add option to export notes to a file.
- [X] Work on the UI
- [ ] Review the recording process to make it more user-friendly



Here are a few suggestions for next steps or improvements you might consider:

1. Session management: You might want to implement session expiration or refresh mechanisms to enhance security.

2. Error handling: Continue to improve error handling and user feedback throughout the application.

3. User profile: Consider adding a user profile page where users can view and edit their information.

4. Security enhancements: Implement CSRF protection and ensure all sensitive routes are properly secured.

5. Testing: Develop a comprehensive test suite to ensure all features work as expected, especially around authentication and authorization.

6. UI/UX improvements: Enhance the user interface and experience based on user feedback.

7. Performance optimization: As your app grows, you might want to optimize database queries and implement caching where appropriate.

If you have any other features you'd like to add or improvements you want to make, feel free to ask. I'm here to help you continue developing and refining your application!


