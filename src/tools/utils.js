/**
 * Chroma Database Utilities
 * A simple JavaScript library for interacting with Chroma DB REST API v2
 */

// Settings
const CHROMA_CONFIG = {
    DEFAULT_HOST: '127.0.0.1',
    DEFAULT_PORT: 9000,
    API_VERSION: 'v2',
    DEFAULT_TENANT: 'default_tenant',
    DEFAULT_DATABASE: 'default_database'
};

class ChromaUtils {
    /**
     * Build the base API URL
     * @param {string} baseUrl - Base URL of the Chroma server 
     * @returns {string} - API base URL with version
     */
    static getApiUrl(baseUrl) {
        return `${baseUrl}/api/${CHROMA_CONFIG.API_VERSION}`;
    }

    /**
     * Get default server URL
     * @returns {string} - Default server URL
     */
    static getDefaultServerUrl() {
        return `http://${CHROMA_CONFIG.DEFAULT_HOST}:${CHROMA_CONFIG.DEFAULT_PORT}`;
    }

    /**
     * Test connection to Chroma server by calling the heartbeat endpoint
     * @param {string} baseUrl - Base URL of the Chroma server
     * @returns {Promise<boolean>} - True if connection is successful
     */
    static async testConnection(baseUrl) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const response = await fetch(`${apiUrl}/heartbeat`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Heartbeat response:', data);
                return true;
            } else {
                console.error('Heartbeat failed:', response.status, response.statusText);
                return false;
            }
        } catch (error) {
            console.error('Connection test failed:', error);
            return false;
        }
    }

    /**
     * List all collections from Chroma server
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {Object} options - Optional parameters (limit, offset, tenant, database)
     * @returns {Promise<Array>} - Array of collection objects
     */
    static async listCollections(baseUrl, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            const params = new URLSearchParams();
            if (options.limit) params.append('limit', options.limit);
            if (options.offset) params.append('offset', options.offset);
            
            const url = `${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections${params.toString() ? '?' + params.toString() : ''}`;
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to list collections: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Collections response:', data);
            
            // Handle different response formats
            return Array.isArray(data) ? data : (data.collections || []);
        } catch (error) {
            console.error('Failed to list collections:', error);
            throw error;
        }
    }

    /**
     * Count collections in Chroma server
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {Object} options - Optional parameters (tenant, database)
     * @returns {Promise<number>} - Number of collections
     */
    static async countCollections(baseUrl, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections_count`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to count collections: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            return data.count || 0;
        } catch (error) {
            console.error('Failed to count collections:', error);
            throw error;
        }
    }

    /**
     * Get a specific collection by name
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {string} collectionId - ID or name of the collection
     * @param {Object} options - Optional parameters (tenant, database)
     * @returns {Promise<Object>} - Collection object
     */
    static async getCollection(baseUrl, collectionId, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections/${encodeURIComponent(collectionId)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get collection: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to get collection:', error);
            throw error;
        }
    }

    /**
     * Get data from a collection
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {string} collectionId - ID or name of the collection
     * @param {Object} options - Query options (ids, where, limit, offset, include, where_document, tenant, database)
     * @returns {Promise<Object>} - Collection data
     */
    static async getCollectionData(baseUrl, collectionId, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const requestBody = {
                ids: options.ids || undefined,
                where: options.where || undefined,
                limit: options.limit || 10,
                offset: options.offset || 0,
                include: options.include || ["metadatas", "documents", "embeddings"],
                where_document: options.where_document || undefined
            };

            // Remove undefined values
            Object.keys(requestBody).forEach(key => {
                if (requestBody[key] === undefined) {
                    delete requestBody[key];
                }
            });

            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections/${encodeURIComponent(collectionId)}/get`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Failed to get collection data: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to get collection data:', error);
            throw error;
        }
    }

    /**
     * Peek at a collection (get first few items) - using get endpoint with limit
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {string} collectionId - ID or name of the collection
     * @param {number} limit - Number of items to peek at (default 10)
     * @param {Object} options - Optional parameters (tenant, database)
     * @returns {Promise<Object>} - Collection data
     */
    static async peekCollection(baseUrl, collectionId, limit = 10, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const requestBody = {
                limit: limit,
                include: ["metadatas", "documents", "embeddings"]
            };

            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections/${encodeURIComponent(collectionId)}/get`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Failed to peek collection: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to peek collection:', error);
            throw error;
        }
    }

    /**
     * Count items in a collection
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {string} collectionId - ID or name of the collection
     * @param {Object} options - Optional parameters (tenant, database)
     * @returns {Promise<number>} - Number of items in the collection
     */
    static async countCollectionItems(baseUrl, collectionId, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections/${encodeURIComponent(collectionId)}/count`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to count collection items: ${response.status} ${response.statusText}`);
            }

            const count = await response.json();
            return count || 0;
        } catch (error) {
            console.error('Failed to count collection items:', error);
            throw error;
        }
    }

    /**
     * Query a collection with similarity search
     * @param {string} baseUrl - Base URL of the Chroma server
     * @param {string} collectionId - ID or name of the collection
     * @param {Object} options - Query options (tenant, database, query_embeddings, query_texts, etc.)
     * @returns {Promise<Object>} - Query results
     */
    static async queryCollection(baseUrl, collectionId, options = {}) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const tenant = options.tenant || CHROMA_CONFIG.DEFAULT_TENANT;
            const database = options.database || CHROMA_CONFIG.DEFAULT_DATABASE;
            
            const requestBody = {
                query_embeddings: options.query_embeddings || undefined,
                query_texts: options.query_texts || undefined,
                n_results: options.n_results || 10,
                where: options.where || undefined,
                where_document: options.where_document || undefined,
                include: options.include || ["metadatas", "documents", "distances"],
                ids: options.ids || undefined
            };

            // Remove undefined values
            Object.keys(requestBody).forEach(key => {
                if (requestBody[key] === undefined) {
                    delete requestBody[key];
                }
            });

            // Ensure we have either query_embeddings or query_texts
            if (!requestBody.query_embeddings && !requestBody.query_texts) {
                throw new Error('Either query_embeddings or query_texts must be provided');
            }

            const response = await fetch(`${apiUrl}/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}/collections/${encodeURIComponent(collectionId)}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`Failed to query collection: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to query collection:', error);
            throw error;
        }
    }

    /**
     * Get server version information
     * @param {string} baseUrl - Base URL of the Chroma server
     * @returns {Promise<Object>} - Server version info
     */
    static async getVersion(baseUrl) {
        try {
            const apiUrl = this.getApiUrl(baseUrl);
            const response = await fetch(`${apiUrl}/version`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get version: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to get version:', error);
            throw error;
        }
    }

    /**
     * Utility function to format metadata for display
     * @param {Object} metadata - Metadata object
     * @returns {string} - Formatted metadata string
     */
    static formatMetadata(metadata) {
        if (!metadata || typeof metadata !== 'object') {
            return 'No metadata';
        }

        return Object.entries(metadata)
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
    }

    /**
     * Utility function to truncate text for display
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length (default 100)
     * @returns {string} - Truncated text
     */
    static truncateText(text, maxLength = 100) {
        if (!text || typeof text !== 'string') {
            return 'No text';
        }

        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    /**
     * Check if a URL is valid
     * @param {string} url - URL to validate
     * @returns {boolean} - True if URL is valid
     */
    static isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch (error) {
            return false;
        }
    }

    /**
     * Handle CORS errors and provide user-friendly messages
     * @param {Error} error - The error object
     * @returns {string} - User-friendly error message
     */
    static handleApiError(error) {
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            return 'Unable to connect to Chroma server. Please check:\n1. Server is running\n2. URL is correct\n3. CORS is enabled on the server';
        }
        
        if (error.message.includes('404')) {
            return 'Endpoint not found. Please check your Chroma server version.';
        }
        
        if (error.message.includes('500')) {
            return 'Internal server error. Please check your Chroma server logs.';
        }
        
        return error.message || 'An unknown error occurred';
    }
}

// Export for use in browsers that support modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChromaUtils;
}

// Make available globally for direct script inclusion
if (typeof window !== 'undefined') {
    window.ChromaUtils = ChromaUtils;
}