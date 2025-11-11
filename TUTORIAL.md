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

## 3. PDF to Text Conversion with Character Mapping

Convert the extracted PDF data to raw text with character position mapping:

```bash
# Convert PDF extraction to raw text with character mapping
bookwyrm pdf-to-text data/heinrich_palaces_pages_1-4.json --verbose
```

This creates two files:
- `data/heinrich_pages_1-4_raw.txt` - Raw text with all PDF text elements joined by newlines
- `data/heinrich_pages_1-4_mapping.json` - Character mapping that links each character position to its bounding box coordinates and page number

You can also specify custom output filenames:

```bash
# Convert with custom output filenames
bookwyrm pdf-to-text data/heinrich_pages_1-4.json \
  --output data/heinrich_text.txt \
  --mapping data/heinrich_char_map.json \
  --verbose
```

## 4. Querying Character Positions

Query specific character ranges to get their bounding box coordinates:

```bash
# Query characters 11948-13000 to see their positions and bounding boxes
bookwyrm pdf-query-range data/Heinrich_palaces_pages_1-4_mapping.json 11948 13000 --verbose
```

This shows you:
- Which pages contain the specified character range
- Bounding box coordinates for each character
- OCR confidence scores
- Sample text from the range

Save the query results to a file:

```bash
# Save bounding box query results to JSON
bookwyrm pdf-query-range data/heinrich_pages_1-4_mapping.json 11948 13000 \
  --output data/character_positions.json \
  --verbose
```

## 5. Phrasal Text Processing

Now let's process a text file to extract meaningful phrases and text spans:

```bash
# Create phrasal analysis of "The Country of the Blind" text
bookwyrm phrasal --file data/country-of-the-blind.txt --output data/country-of-the-blind-phrases.jsonl
```

This generates a JSONL file with text chunks and their positional information, suitable for further analysis.

## 6. Text Summarization

Let's create summaries from the phrasal data we just generated. BookWyrm supports both basic summarization and structured output using Pydantic models.

### Basic Summarization

```bash
# Generate a basic summary from the Country of the Blind phrases
bookwyrm summarize data/country-of-the-blind-phrases.jsonl --output data/country-of-the-blind-summary.json --max-tokens 500 --verbose
```

### Structured Literary Analysis with Pydantic Models

For more structured analysis, we can use the `Summary` class to extract specific literary elements:

```bash
# Generate structured literary analysis using the Summary model
bookwyrm summarize data/country-of-the-blind-phrases.jsonl \
  --model-class-file data/summary.py \
  --model-class-name Summary \
  --model-strength smart \
  --output data/country-structured-summary.json \
  --verbose
```

This produces a structured JSON output with specific fields like:
- `title`: The work's title
- `author`: Author information
- `date_of_publication`: Publication date
- `plot`: Comprehensive plot summary
- `timeline`: Temporal setting and chronology
- `important_characters`: Key characters and figures

The structured output will look like:
```json
{
  "summary": {
    "title": "The Country of the Blind",
    "author": "H.G. Wells",
    "date_of_publication": "1904-01-01",
    "plot": "A mountaineer named Nunez discovers an isolated valley...",
    "timeline": "Early 20th century, set in an isolated Andean valley...",
    "important_characters": ["Nunez", "Medina-saroté", "Yacob", "The Elders"]
  },
  "subsummary_count": 3,
  "levels_used": 2,
  "total_tokens": 1250,
  "source_file": "data/country-of-the-blind-phrases.jsonl",
  "model_used": "Summary"
}
```

### Advanced Model Strengths

You can also use different model strengths for varying levels of analysis quality:

```bash
# High-quality literary analysis with the wise model
bookwyrm summarize data/country-of-the-blind-phrases.jsonl \
  --model-class-file data/summary.py \
  --model-class-name Summary \
  --model-strength wise \
  --max-tokens 2000 \
  --output data/country-detailed-analysis.json
```

## 7. Citation Finding

Finally, let's find specific citations related to life-threatening situations in the story:

```bash
# Find citations about life-threatening situations the protagonist faces
bookwyrm cite data/country-of-the-blind-phrases.jsonl --question "Where does the protagonist experience life threatening situations?" --output data/protagonist-dangers.json --verbose --long
```

This searches through the text chunks to find relevant passages that answer the specific question about dangerous situations.

## Complete PDF Processing Workflow

Here's a complete workflow for processing a PDF from extraction to position queries:

```bash
# Step 1: Extract PDF structure
bookwyrm extract-pdf data/Heinrich_palaces.pdf --start-page 1 --num-pages 4 --output data/heinrich_extracted.json

# Step 2: Convert to raw text with character mapping
bookwyrm pdf-to-text data/heinrich_extracted.json --verbose

# Step 3: Query specific character ranges
bookwyrm pdf-query-range data/heinrich_extracted_mapping.json 0 100 --verbose

# Step 4: Query a larger range and save results
bookwyrm pdf-query-range data/heinrich_extracted_mapping.json 1000 2000 --output data/positions_1000-2000.json
```

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

### Custom Prompts for Summarization
Instead of using Pydantic models, you can provide custom prompts:

```bash
# Use custom prompts for specialized analysis
bookwyrm summarize data/country-of-the-blind-phrases.jsonl \
  --chunk-prompt "Extract key themes, symbols, and literary devices from this text" \
  --summary-prompt "Create a comprehensive literary analysis focusing on themes, symbolism, and narrative techniques" \
  --model-strength clever \
  --output data/country-literary-analysis.json
```

### Debug Mode
Use `--include-debug` to see intermediate summaries:

```bash
# Run with debug information to see intermediate summaries
bookwyrm summarize data/country-of-the-blind-phrases.jsonl \
  --model-class-file data/summary.py \
  --model-class-name Summary \
  --include-debug \
  --output data/country-debug-summary.json
```

## Expected Output Files

After running these commands, you should have:
- `data/heinrich_pages_1-4.json` - Structured PDF data with bounding boxes
- `data/heinrich_pages_1-4_raw.txt` - Raw text extracted from PDF
- `data/heinrich_pages_1-4_mapping.json` - Character position to bounding box mapping
- `data/character_positions.json` - Query results for specific character ranges
- `data/country-of-the-blind-phrases.jsonl` - Phrasal analysis
- `data/country-of-the-blind-summary.json` - Basic text summary
- `data/country-structured-summary.json` - Structured literary analysis using Summary model
- `data/country-detailed-analysis.json` - High-quality structured analysis
- `data/protagonist-dangers.json` - Citation results

These files demonstrate the full pipeline from raw documents to structured insights using the BookWyrm API, including the ability to map text positions back to their original locations in PDF documents.

## Use Cases for Character Mapping

The character mapping functionality enables several powerful use cases:

1. **Citation Highlighting**: Find citations in text, then highlight the exact regions in the original PDF
2. **Search Result Visualization**: Show users exactly where search results appear in the document
3. **Annotation Systems**: Allow users to annotate text and map annotations back to PDF coordinates
4. **Quality Analysis**: Analyze OCR confidence scores for specific text regions
5. **Layout Analysis**: Understand how text flows across pages and identify reading order
