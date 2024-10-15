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
  - `created_at`: The timestamp when the note was created.

## To do

- Add a column to the `notes` table to store the transcription cost.
- Add option to export notes to a file.
- Work on the UI
- Review the recording process to make it more user-friendly


