import React, { useState, useEffect } from "react";
import { Plus, ChevronDown, ChevronUp, Edit, Trash } from "lucide-react";
import axios from "axios";

interface EventItem {
  id: number;          // event_ID
  title: string;       // event_name
  category: string;    // event_category
  date: string;        // YYYY-MM-DD from start_time
  startTime: string;   // HH:mm from start_time
  endTime: string;     // HH:mm from end_time
  location: string;
  media_urls: string;
  description: string;
  organizer_id: number; // organizer_ID
}

const EventsWidget: React.FC = () => {
  const [showForm, setShowForm] = useState(false);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);

  const [formData, setFormData] = useState<Omit<EventItem, "id"> & { id?: number }>({
    title: "",
    category: "",
    date: "",
    startTime: "",
    endTime: "",
    location: "",
    media_urls: "",
    description: "",
    organizer_id: 0,
  });

  // Fetch events from backend and map DB fields
  const fetchEvents = async () => {
    try {
      const res = await axios.get("http://localhost:5000/events");

      const mapped = res.data.map((ev: any) => {
        // ⚡ Don't use new Date() → it causes timezone shifts
        const startParts = ev.start_time ? ev.start_time.split(" ") : ["", ""];
        const endParts = ev.end_time ? ev.end_time.split(" ") : ["", ""];

        return {
          id: ev.event_id,
          title: ev.event_name,
          category: ev.event_category,
          date: startParts[0] || "",
          startTime: startParts[1] ? startParts[1].slice(0, 5) : "",
          endTime: endParts[1] ? endParts[1].slice(0, 5) : "",
          location: ev.location || "",
          media_urls: ev.media_urls || "",
          description: ev.description || "",
          organizer_id: ev.organizer_id ?? 0,
        };
      });

      setEvents(mapped);
    } catch (err) {
      console.error("Error fetching events:", err);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.title) {
      alert("Title is required.");
      return;
    }
    if (formData.organizer_id <= 0) {
      alert("Organizer ID must be a positive number.");
      return;
    }

    // ✅ Build proper timestamps from strings without Date() conversion
    const startISO =
      formData.date && formData.startTime
        ? `${formData.date}T${formData.startTime}:00`
        : null;
    const endISO =
      formData.date && formData.endTime
        ? `${formData.date}T${formData.endTime}:00`
        : null;

    const payload = {
      event_name: formData.title,
      event_category: formData.category,
      start_time: startISO,
      end_time: endISO,
      location: formData.location,
      description: formData.description,
      media_urls: formData.media_urls,
      organizer_id: formData.organizer_id,
    };

    try {
      if (formData.id && events.some((ev) => ev.id === formData.id)) {
        // Edit existing
        await axios.put(`http://localhost:5000/events/${formData.id}`, payload);
      } else {
        // Add new
        await axios.post("http://localhost:5000/events", payload);
      }
      await fetchEvents();
    } catch (err) {
      console.error("Error saving event:", err);
    }

    setFormData({
      title: "",
      category: "",
      date: "",
      startTime: "",
      endTime: "",
      location: "",
      media_urls: "",
      description: "",
      organizer_id: 0,
    });
    setShowForm(false);
  };

  const handleEdit = (event: EventItem) => {
    setFormData({
      id: event.id,
      title: event.title,
      category: event.category,
      date: event.date,
      startTime: event.startTime,
      endTime: event.endTime,
      location: event.location,
      media_urls: event.media_urls,
      description: event.description,
      organizer_id: event.organizer_id,
    });
    setShowForm(true);
  };

  const handleDelete = async (id: number) => {
    try {
      await axios.delete(`http://localhost:5000/events/${id}`);
      fetchEvents();
    } catch (err) {
      console.error("Error deleting event:", err);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <button
          onClick={() => {
            setFormData({
              title: "",
              category: "",
              date: "",
              startTime: "",
              endTime: "",
              location: "",
              media_urls: "",
              description: "",
              organizer_id: 0,
            });
            setShowForm(true);
          }}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          <Plus size={16} /> Add Event
        </button>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-xl shadow-lg w-full max-w-lg max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-bold mb-4">
              {formData.id ? "Edit Event" : "Add Event"}
            </h3>
            <form onSubmit={handleSubmit} className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700">Title</label>
                <input
                  type="text"
                  value={formData.title}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Category</label>
                <input
                  type="text"
                  value={formData.category}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Date</label>
                <input
                  type="date"
                  value={formData.date}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Start Time</label>
                <input
                  type="time"
                  value={formData.startTime}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, startTime: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">End Time</label>
                <input
                  type="time"
                  value={formData.endTime}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, endTime: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Location</label>
                <input
                  type="text"
                  value={formData.location}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Media URLs</label>
                <input
                  type="text"
                  value={formData.media_urls}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, media_urls: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  value={formData.description}
                  className="w-full border p-2 rounded"
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Organizer ID</label>
                <input
                  type="number"
                  value={formData.organizer_id || ""}
                  min={1}
                  className="w-full border p-2 rounded"
                  onChange={(e) => {
                    const val = e.target.value;
                    setFormData({ ...formData, organizer_id: val ? parseInt(val) : 0 });
                  }}
                />
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="px-4 py-2 bg-gray-200 rounded"
                >
                  Cancel
                </button>
                <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded">
                  Save
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Events List */}
      <div className="space-y-3">
        {events.map((ev) => (
          <div key={ev.id} className="border rounded-lg bg-white shadow-sm">
            <button
              onClick={() => setExpanded(expanded === ev.id ? null : ev.id)}
              className="w-full flex justify-between items-center px-4 py-3 text-left font-medium hover:bg-gray-50"
            >
              {ev.title} (ID: {ev.id}){" "}
              <span className="text-gray-500 text-sm">Organizer: {ev.organizer_id}</span>
              {expanded === ev.id ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
            {expanded === ev.id && (
              <div className="px-4 pb-4 space-y-2">
                <p><span className="font-semibold">Event ID:</span> {ev.id}</p>
                <p><span className="font-semibold">Category:</span> {ev.category}</p>
                <p><span className="font-semibold">Date:</span> {ev.date}</p>
                <p><span className="font-semibold">Time:</span> {ev.startTime} - {ev.endTime}</p>
                <p><span className="font-semibold">Location:</span> {ev.location}</p>
                <p><span className="font-semibold">Media URLs:</span> {ev.media_urls}</p>
                <p><span className="font-semibold">Description:</span> {ev.description}</p>
                <p><span className="font-semibold">Organizer ID:</span> {ev.organizer_id}</p>
                <div className="flex gap-3 mt-2">
                  <button
                    onClick={() => handleEdit(ev)}
                    className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600"
                  >
                    <Edit size={14} className="inline-block mr-1" /> Edit
                  </button>
                  <button
                    onClick={() => handleDelete(ev.id)}
                    className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600"
                  >
                    <Trash size={14} className="inline-block mr-1" /> Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default EventsWidget;
