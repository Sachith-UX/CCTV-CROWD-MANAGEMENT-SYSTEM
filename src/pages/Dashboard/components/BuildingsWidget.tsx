import React, { useState, useEffect } from "react";
import axios from "axios";
import { Plus, ChevronDown, ChevronUp, Edit, Trash } from "lucide-react";

interface Building {
  id: string; // Local ID for frontend state
  building_id: number;
  zone_id: number;
  building_name: string;
  description: string;
}

const BuildingsWidget: React.FC = () => {
  const [showForm, setShowForm] = useState(false);
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  const [formData, setFormData] = useState<Building>({
    id: "",
    building_id: 0,
    zone_id: 0,
    building_name: "",
    description: "",
  });

  // Fetch all buildings on component mount
  useEffect(() => {
    axios
      .get("http://localhost:5000/buildings")  // Fetch building data
      .then((res) => {
        const formatted = res.data.map((b: any) => ({
          ...b,
          id: b.building_id.toString(), // Use building_id as unique id
        }));
        setBuildings(formatted);
      })
      .catch((err) => console.error("Error fetching buildings:", err));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!formData.building_name) return;
    if (formData.building_id < 0 || formData.zone_id < 0) {
      alert("IDs cannot be negative");
      return;
    }

    try {
      if (formData.id) {
        // Update existing building
        await axios.put(
          `http://localhost:5000/buildings/${formData.building_id}`,
          {
            zone_id: formData.zone_id,
            building_name: formData.building_name,
            description: formData.description,
          }
        );
        // Update the buildings list in the UI
        setBuildings((prev) =>
          prev.map((b) => (b.id === formData.id ? formData : b))
        );
      } else {
        // Create new building
        const res = await axios.post("http://localhost:5000/buildings", {
          zone_id: formData.zone_id,
          building_name: formData.building_name,
          description: formData.description,
        });

        if (res.status === 200) {
          const newBuilding = {
            ...formData,
            id: Date.now().toString(),  // Generate temporary ID
            building_id: res.data.building_id, // Get real building_id from response
          };
          setBuildings((prev) => [...prev, newBuilding]);
        }
      }
    } catch (err) {
      console.error("Error saving building:", err);
    }

    // Reset form after submission
    setFormData({
      id: "",
      building_id: 0,
      zone_id: 0,
      building_name: "",
      description: "",
    });
    setShowForm(false);
  };

  const handleEdit = (building: Building) => {
    setFormData(building);
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    const building = buildings.find((b) => b.id === id);
    if (!building) return;

    try {
      await axios.delete(`http://localhost:5000/buildings/${building.building_id}`);
      // Remove deleted building from state
      setBuildings((prev) => prev.filter((b) => b.id !== id));
    } catch (err) {
      console.error("Error deleting building:", err);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <button
          onClick={() => setShowForm(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          <Plus size={16} /> Add Building
        </button>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-xl shadow-lg w-full max-w-lg max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-bold mb-4">
              {formData.id ? "Edit Building" : "Add Building"}
            </h3>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block font-medium mb-1">Building ID</label>
                <input
                  type="number"
                  min="0"
                  value={formData.building_id}
                  className="w-full border p-2 rounded"
                  disabled={!!formData.id} // Disable when editing
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      building_id: Number(e.target.value),
                    })
                  }
                />
              </div>

              <div>
                <label className="block font-medium mb-1">Zone ID</label>
                <input
                  type="number"
                  min="0"
                  value={formData.zone_id}
                  className="w-full border p-2 rounded"
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      zone_id: Number(e.target.value),
                    })
                  }
                />
              </div>

              <div>
                <label className="block font-medium mb-1">Building Name</label>
                <input
                  type="text"
                  value={formData.building_name}
                  className="w-full border p-2 rounded"
                  onChange={(e) =>
                    setFormData({ ...formData, building_name: e.target.value })
                  }
                />
              </div>

              <div>
                <label className="block font-medium mb-1">Description</label>
                <textarea
                  value={formData.description}
                  className="w-full border p-2 rounded"
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                ></textarea>
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowForm(false)}
                  className="px-4 py-2 bg-gray-200 rounded"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded"
                >
                  Save
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Buildings List */}
      <div className="space-y-3">
        {buildings.map((b) => (
          <div key={b.id} className="border rounded-lg bg-white shadow-sm">
            <button
              onClick={() => setExpanded(expanded === b.id ? null : b.id)}
              className="w-full flex justify-between items-center px-4 py-3 text-left font-medium hover:bg-gray-50"
            >
              {b.building_name}
              {expanded === b.id ? (
                <ChevronUp size={18} />
              ) : (
                <ChevronDown size={18} />
              )}
            </button>
            {expanded === b.id && (
              <div className="px-4 pb-4 space-y-2">
                <p>
                  <span className="font-semibold">Building ID:</span>{" "}
                  {b.building_id}
                </p>
                <p>
                  <span className="font-semibold">Zone ID:</span> {b.zone_id}
                </p>
                <p>
                  <span className="font-semibold">Description:</span>{" "}
                  {b.description}
                </p>

                <div className="flex gap-3 mt-2">
                  <button
                    onClick={() => handleEdit(b)}
                    className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600"
                  >
                    <Edit size={14} className="inline-block mr-1" /> Edit
                  </button>
                  <button
                    onClick={() => handleDelete(b.id)}
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

export default BuildingsWidget;
