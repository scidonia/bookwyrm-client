# BookWyrm CLI Tutorial

This tutorial demonstrates the key capabilities of the BookWyrm CLI through practical examples using sample data files.

## 1. File Classification

First, let's classify a PDF file to understand its content type and structure:

```bash
# Classify the Heinrich palaces PDF to understand its content
bookwyrm classify data/Heinrich_palaces.pdf
```

This will analyze the PDF and return classification information including file type, content analysis, and structural metadata.

## 2. PDF Structure Extraction

Next, let's extract structured data from specific pages of the PDF:

```bash
# Extract structured JSON data from pages 1-4 of the Heinrich palaces PDF
bookwyrm extract-pdf data/Heinrich_palaces.pdf --start-page 1 --num-pages 4 --output data/heinrich_pages_1-4.json
```

This creates a JSON file containing the structured text, bounding boxes, and layout information for the specified pages.

## 3. Phrasal Text Processing

Now let's process a text file to extract meaningful phrases and text spans:

```bash
# Create phrasal analysis of "The Country of the Blind" text
bookwyrm phrasal --file data/country-of-the-blind.txt --output data/country-of-the-blind-phrases.jsonl --chunk-size 1000
```

This generates a JSONL file with text chunks and their positional information, suitable for further analysis.

## 4. Text Summarization

Let's create a summary from the phrasal data we just generated:

```bash
# Generate a summary from the Country of the Blind phrases
bookwyrm summarize data/country-of-the-blind-phrases.jsonl --output data/country-of-the-blind-summary.json --max-tokens 500 --verbose
```

This produces a structured summary with key insights from the text.

## 5. Citation Finding

Finally, let's find specific citations related to life-threatening situations in the story:

```bash
# Find citations about life-threatening situations the protagonist faces
bookwyrm cite data/country-of-the-blind-phrases.jsonl --question "Where does the protagonist experience life threatening situations?" --output data/protagonist-dangers.json --verbose --long
```

This searches through the text chunks to find relevant passages that answer the specific question about dangerous situations.

## Additional Options

### Streaming Output
For real-time processing feedback, add the `--stream` flag to most commands:

```bash
# Stream the summarization process
bookwyrm summarize data/country-of-the-blind-phrases.jsonl --stream --verbose
```

### Multiple Questions
You can ask multiple citation questions at once:

```bash
# Ask multiple questions about the story
bookwyrm cite data/country-of-the-blind-phrases.jsonl \
  --question "What are the main conflicts in the story?" \
  --question "How does the protagonist adapt to his environment?" \
  --question "What role does blindness play in the narrative?" \
  --verbose
```

### Debug Mode
Use `--debug` to see detailed API interactions:

```bash
# Run with debug information
bookwyrm summarize data/country-of-the-blind-phrases.jsonl --debug --max-tokens 200
```

## Expected Output Files

After running these commands, you should have:
- `data/heinrich_pages_1-4.json` - Structured PDF data
- `data/country-of-the-blind-phrases.jsonl` - Phrasal analysis
- `data/country-of-the-blind-summary.json` - Text summary
- `data/protagonist-dangers.json` - Citation results

These files demonstrate the full pipeline from raw documents to structured insights using the BookWyrm API.
