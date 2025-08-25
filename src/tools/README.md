# Chroma Database Browser

A simple, standalone HTML/JavaScript tool for browsing and exploring Chroma vector database collections. No server required - just open the HTML file in your browser!

## Features

- **Simple Setup**: Pure HTML/JS - no compilation or server needed
- **Collection Browsing**: View all collections in your Chroma database
- **Data Exploration**: Browse documents, metadata, and embeddings
- **Connection Testing**: Built-in heartbeat to test server connectivity
- **Responsive Design**: Clean, modern interface that works on desktop and mobile
- **CORS-Ready**: Handles CORS issues with helpful error messages
- **API v2 Compatible**: Uses the latest Chroma REST API v2 endpoints

## Quick Start

1. Make sure your Chroma server is running (default: `http://127.0.0.1:9000`)
2. Open `index.html` in your web browser
3. Click "Connect" to connect to your Chroma server
4. Click "Load Collections" to see available collections
5. Click on any collection to explore its data

## Files

- **`index.html`** - Main browser interface with responsive design
- **`utils.js`** - JavaScript utilities for Chroma API v2 interactions
- **`README.md`** - This documentation file

## Chroma Server Requirements

This tool connects to a Chroma server via REST API v2. Ensure your server:

- Is running and accessible at the specified URL
- Has CORS enabled for browser access
- Uses the Chroma REST API v2 endpoints (`/api/v2/`)

### Configuration
The tool is configured for:
- **Default Host**: `127.0.0.1`
- **Default Port**: `9000`
- **API Version**: `v2`

You can modify these settings in the `CHROMA_CONFIG` section at the top of `utils.js`.

## Supported Operations

### Connection Management
- Test connection with heartbeat endpoint
- Display connection status with visual indicators
- Handle connection errors gracefully

### Collection Operations
- List all collections with metadata
- Count total collections
- Get detailed collection information
- Browse collection data with pagination

### Data Exploration
- **Get**: Retrieve specific items by ID or use filters
- **Peek**: Quick preview of collection contents
- **Metadata Viewing**: Display all metadata fields
- **Document Browsing**: View document text with truncation
- **Embedding Info**: Show embedding dimensions

## Browser Compatibility

Works with all modern browsers that support:
- ES6+ JavaScript features
- Fetch API
- CSS Grid and Flexbox

Tested with Chrome, Firefox, Safari, and Edge.

## Troubleshooting

### Connection Issues
- **"Unable to connect"**: Check if Chroma server is running
- **CORS Errors**: Enable CORS on your Chroma server
- **404 Errors**: Verify your Chroma server supports API v2

### Common Solutions
1. Start Chroma with CORS enabled: `chroma run --host 0.0.0.0 --port 9000`
2. Check server logs for error details
3. Verify the server URL is correct (no trailing slash)

### Chroma Server Setup
If you need to set up a Chroma server:

```bash
# Install Chroma
pip install chromadb

# Run server with CORS enabled
chroma run --host 0.0.0.0 --port 9000
```

## Usage Examples

### Basic Browsing
1. Connect to your Chroma server
2. Load collections to see what's available
3. Click on a collection to explore its data
4. Use the limit/offset controls for pagination

### Data Exploration
- View document text and metadata for each item
- Check embedding dimensions and vector information
- Use the "Peek" feature for quick data sampling

## API Endpoints Used

The tool interacts with these Chroma REST API v2 endpoints:

- `GET /api/v2/heartbeat` - Test connection
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections` - List collections
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections/{collection_id}` - Get collection details
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection_id}/get` - Get collection data
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections/{collection_id}/count` - Count items
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection_id}/query` - Query collection
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections_count` - Count total collections
- `GET /api/v2/version` - Get server version

## Security Notes

- This tool makes direct HTTP requests to your Chroma server
- Ensure your Chroma server is secured appropriately
- Consider network security when exposing Chroma servers
- No authentication is implemented - relies on server-side security

## Contributing

This is a simple, standalone tool. To modify:

1. Edit `index.html` for UI changes
2. Modify `utils.js` for API functionality
3. Update `CHROMA_CONFIG` settings as needed
4. Test with your Chroma server setup

## License

This tool is provided as-is for exploring Chroma databases. Check the parent project license for specific terms.